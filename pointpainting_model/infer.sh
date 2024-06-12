#! /bin/bash

python3 infer.py --waymo_path /work/waymo/kitti_format/ --lidar_model_path /work/model/PointPillar_epoch_48.pth  --segmentation_model_path /work/model/Deeplabv3plus_RN50.pth --cam_sync

