import argparse
import numpy as np
import os
import torch
import copy
from collections import namedtuple
from tqdm import tqdm
import time

from dataset import Waymo, get_dataloader
from pointpainting import PointPainting


def convert_calib(calib, cuda):
    result = {}
    result['R0_rect'] = torch.from_numpy(calib['R0_rect'])
    for i in range(5):
        result['P' + str(i)] = torch.from_numpy(calib['P' + str(i)])
        result['Tr_velo_to_cam_' + str(i)] = torch.from_numpy(calib['Tr_velo_to_cam_' + str(i)])
    return change_calib_device(result, cuda)

def change_calib_device(calib, cuda):
    result = {}
    if cuda:
        device = 'cuda'
    else:
        device='cpu'
    result['R0_rect'] = calib['R0_rect'].to(device=device, dtype=torch.float)
    for i in range(5):
        result['P' + str(i)] = calib['P' + str(i)].to(device=device, dtype=torch.float)
        result['Tr_velo_to_cam_' + str(i)] = calib['Tr_velo_to_cam_' + str(i)].to(device=device, dtype=torch.float)
    return result

def run_infer(args):
    val_dataset = Waymo(data_root=args.waymo_path,
                        split='val', 
                        cam_sync=args.cam_sync, 
                        inference=True, 
                        painted=False)
    val_dataloader, _ = get_dataloader(dataset=val_dataset, 
                                       batch_size=1, 
                                       num_workers=args.num_workers,
                                       rank=0,
                                       world_size=1,
                                       shuffle=False)
    
    CLASSES = Waymo.CLASSES
    LABEL2CLASSES = {v:k for k, v in CLASSES.items()}

    target_device = 'cuda' if (torch.cuda.is_available() and not args.no_cuda) else 'cpu'

    proj_mat_path = Path(args.projection_matrix_npy_path)
    assert proj_mat_path.is_file(), "Need valid projection matrix file for lidar detection network"
    proj_mat = np.load(str(proj_mat_path))

    PointPaintingModel = PointPainting(args.segmentation_model_path,
                                       args.lidar_model_path,
                                       args.num_classes,
                                       args.cam_sync,
                                       proj_mat,
                                       target_device)

    print('Starting Inference Task...')
    result_dict = dict()
    with torch.inference_mode():
        for i, data_dict in enumerate(tqdm(val_dataloader)):
            # Prep the input tensors
            data_dict['batched_calib_info'][0] = convert_calib(data_dict['batched_calib_info'][0], target_device)
            data_dict['batched_pts'][0].to(target_device)
            for i in range(len(data_dict['batched_images'][0])):
                data_dict['batched_images'][0][i] = data_dict['batched_images'][0][i].to(target_device)
            for key in data_dict:
                for j, item in enumerate(data_dict[key]):
                    if torch.is_tensor(item):
                        data_dict[key][j] = data_dict[key][j].to(target_device)
            
            batched_pts = data_dict['batched_pts']
            batched_gt_bboxes = data_dict['batched_gt_bboxes']
            batched_labels = data_dict['batched_labels']

            start_time = time.perf_counter()
            batched_results = PointPaintingModel(
                                data_dict['batched_images'][0],
                                batched_pts[0],
                                data_dict['batched_calib_info'][0])
            end_time = time.perf_counter()
            iter_time = end_time - start_time

            assert len(batched_results) == 1, "Expecting only one output tensor"
            results_dict[i] = {
                'results': batched_results[0].to('cpu'),
                'latency': iter_time,
                'ground_truth': {
                    'lidar_bboxes': data_dict['batched_gt_bboxxes'][0],
                    'lables': data_dict['batched_labels'][0]
                }
            }
    
    return result_dict
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--lidar_model_path', help='Path to lidar (PointPillar) model', required=True)
    parser.add_argument('--segmentation_model_path', help='Path to segmentation (DeepLabV3plus_RN50) model', required=True)
    parser.add_argument('--projection_matrix_npy_path', help='Path to projection matrix numpy file', required=True)
    parser.add_argument('--num_classes', type=int, default=3, help='Number of classes for 3D detection')
    parser.add_argument('--cam_sync', action='store_true', help='Use only objects visible to a camera')
    parser.add_argument('--waymo_path', help='Path to data root of waymo dataset')
    parser.add_argument('--no_cuda', action='store_true', help='Do not use CUDA even if it is available')

    args = parser.parse_args()

    results = run_infer(args)
    eval_results = run_eval(results)
    