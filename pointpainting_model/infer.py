import argparse
import torch
import numpy as np
import time
import copy

from dataset import Waymo, get_dataloader
from pointpainting import PointPainting
from pathlib import Path


def run_infer(args):

    def convert_calib(calib, device='cpu'):
        result = {}
        result['R0_rect'] = torch.from_numpy(calib['R0_rect']).to(device, dtype=torch.float)
        for i in range(5):
            result['P' + str(i)] = torch.from_numpy(calib['P' + str(i)]).to(device, dtype=torch.float)
            result['Tr_velo_to_cam_' + str(i)] = \
                torch.from_numpy(calib['Tr_velo_to_cam_' + str(i)]).to(device, dtype=torch.float)
        return result

    val_dataset = Waymo(data_root=args.data_root,
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

    segmenter_checkpoint_path = Path(args.segmentor).absolute()
    detector_checkpoint_path = Path(args.lidar_detector).absolute()
    assert segmenter_checkpoint_path.is_file(), "Need valid checkpoint file for segmentation network"
    assert detector_checkpoint_path.is_file(), "Need valid checkpoint file for lidar detection network"


    PointPaintingModel = PointPainting(str(segmenter_checkpoint_path),
                                       str(detector_checkpoint_path),
                                       args.num_classes,
                                       args.cam_sync,
                                       target_device)

    results_dict = dict()
    input_img_shape = (1, 3, 1280, 1920)
    calib_names = ['Tr_velo_to_cam_0', 'Tr_velo_to_cam_1', 'Tr_velo_to_cam_2', 'Tr_velo_to_cam_3', 'Tr_velo_to_cam_4',
                   'P0', 'P1', 'P2', 'P3', 'P4', 'R0_rect']
    total = len(val_dataloader)
    print(f'Starting Inference Task for {total} requests...')
    with torch.inference_mode():
        for i, data_dict in enumerate(val_dataloader):
            # Prep the input tensors
            data_dict['batched_calib_info'][0] = convert_calib(data_dict['batched_calib_info'][0], target_device)
            data_dict['batched_pts'][0].to(target_device)
            for j in range(len(data_dict['batched_images'][0])):
                img = data_dict['batched_images'][0][j].to(target_device)
                if img.shape < input_img_shape:
                    padding = [input_img_shape[k] - img.shape[k] for k in range(len(input_img_shape))]
                    padder = torch.nn.ZeroPad2d(padding)
                    img = padder(img)
                assert img.shape == input_img_shape, "Illegal input image shape"
                data_dict['batched_images'][0][j] = img
            for key in data_dict:
                for j, item in enumerate(data_dict[key]):
                    if torch.is_tensor(item):
                        data_dict[key][j] = data_dict[key][j].to(target_device)

            calib_list = [data_dict['batched_calib_info'][0][n] for n in calib_names]
            x = torch.cat(data_dict['batched_images'][0], dim=0).to(target_device)
            y = torch.Tensor(data_dict['batched_pts'][0]).to(target_device)
            z = torch.stack(calib_list).to(target_device)
            
            # FIXME: should I use perf_counter_ns()? 
            start_time = time.perf_counter()
            batched_results = PointPaintingModel(x, y, z)
            end_time = time.perf_counter()
            iter_time = end_time - start_time

            assert len(batched_results) == 1, "Expecting only one output tensor"
            results_dict[i] = {
                'results': copy.deepcopy(batched_results[0]),
                'latency': iter_time,
                'ground_truth': {
                    'lidar_bboxes': copy.deepcopy(data_dict['batched_gt_bboxes'][0]),
                    'lables': copy.deepcopy(data_dict['batched_labels'][0]),
                }
            }
            print(f"{i+1}/{total} took {iter_time} sec")
            if i > 10:
                break
    
    # TORCH to ONNX export
    torch_model = PointPaintingModel
    torch_input_x = x
    torch_input_y = y
    torch_input_z = z
    onnx_program = torch.onnx.dynamo_export(torch_model, torch_input_x, torch_input_y, torch_input_z)
    onnx_program.save("pp.onnx")

    return results_dict
    
def run_eval(results):
    # get latency distribution
    latencies = [i['latency'] for i in results.values()]
    latencies = np.array(latencies)
    mean_lat = latencies.mean()
    throughput = 1. / mean_lat
    print(f"Perf: mean latency [s] = {mean_lat}, samples/sec = {throughput}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--lidar_detector', help='Path to lidar (PointPillar) model', required=True)
    parser.add_argument('--segmentor', help='Path to segmentation (DeepLabV3plus_RN50) model', required=True)
    parser.add_argument('--num_classes', type=int, default=3, help='Number of classes for 3D detection')
    parser.add_argument('--cam_sync', action='store_true', help='Use only objects visible to a camera')
    parser.add_argument('--data_root', help='Path to data root of waymo dataset')
    parser.add_argument('--no_cuda', action='store_true', help='Do not use CUDA even if it is available')
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()

    results = run_infer(args)
    eval_results = run_eval(results)
    
