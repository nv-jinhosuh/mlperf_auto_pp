import torch
from pathlib import Path

from deeplabv3plus import DeepLabV3Plus_ResNet50
from pointpainter import PointPainter
from pointpillars import PointPillars


class PointPainting(nn.Model):
    def __init__(self, 
                 segmentation_model_path,
                 lidar_model_path,
                 num_classes,
                 cam_sync,
                 proj_mat,
                 target_device = 'cpu'):
        super().__init__()

        segmenter_checkpoint_path = Path(segmentation_model_path).absolute()
        detector_checkpoint_path = Path(lidar_model_path).absolute()
        assert segmenter_checkpoint_path.is_file(), "Need valid checkpoint file for segmentation network"
        assert detector_checkpoint_path.is_file(), "Need valid checkpoint file for lidar detection network"
        
        self.device = target_device
        self.proj_mat = proj_mat

        self.segmenter = DeepLabV3Plus_ResNet50(str(segmenter_checkpoint_path), self.device)
        self.painter = PointPainter(cam_sync, self.proj_mat, self.device)
        self.detector = PointPillars(nclasses=num_classes, target_device=self.device)
    
    @torch.no_grad()
    def forward(self, x, y):
        '''
        x: input image captured from cameras, batchsize = number of cameras
        y: input point cloud captured from lidar, batchsize = 1
        '''

        result = self.segmenter(x)
        result = self.painter(result, y)
        result = self.detector(result)

        return result
