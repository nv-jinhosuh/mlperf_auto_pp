#! /bin/bash

python infer.py --data_root /work/waymo/kitti_format --lidar_detector /work/models/pointpainting_ep36.pth --segmentor /work/models/deeplabv3plus_rn50_waaymo.pth --cam_sync
