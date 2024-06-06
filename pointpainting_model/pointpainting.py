import torch
import torch.nn as nn

from deeplabv3plus import DeepLabV3Plus_ResNet50
from pointpainter import PointPainter
from pointpillars import PointPillars


class PointPainting(nn.Module):
    def __init__(self, 
                 segmentation_model,
                 lidar_model,
                 num_classes,
                 cam_sync,
                 target_device = 'cpu'):
        super().__init__()

        self.device = target_device

        self.segmenter = DeepLabV3Plus_ResNet50(segmentation_model, self.device)
        self.painter = PointPainter(cam_sync, self.device)
        self.detector = PointPillars(nclasses=num_classes, checkpoint=lidar_model, 
                                     target_device=self.device)
    
    @torch.no_grad()
    def forward(self, x, y, z):
        '''
        x: input image captured from cameras, batchsize = number of cameras
        y: input point cloud captured from lidar, batchsize = 1
        z: projection matrix
        '''

        result = self.segmenter(x)
        result = self.painter(result, y, z)
        result = self.detector(result)

        return result
