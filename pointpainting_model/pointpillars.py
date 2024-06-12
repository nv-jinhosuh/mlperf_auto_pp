import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ops import Voxelization, nms_cuda


class Anchors(nn.Module):
    def __init__(self, ranges, sizes, rotations, target_device='cpu'):
        super().__init__()
        assert len(ranges) == len(sizes)
        self.ranges = ranges
        self.sizes = sizes
        self.rotations = rotations
        self.device = torch.device(target_device)

    def get_anchors(self, feature_map_size, anchor_range, anchor_size, rotations):
        '''
        feature_map_size: (y_l, x_l)
        anchor_range: [x1, y1, z1, x2, y2, z2]
        anchor_size: [w, l, h]
        rotations: [0, 1.57]
        return: shape=(y_l, x_l, 2, 7)
        '''
        x_centers = torch.linspace(anchor_range[0], anchor_range[3], feature_map_size[1] + 1, device=self.device)
        y_centers = torch.linspace(anchor_range[1], anchor_range[4], feature_map_size[0] + 1, device=self.device)
        z_centers = torch.linspace(anchor_range[2], anchor_range[5], 1 + 1, device=self.device)

        x_shift = (x_centers[1] - x_centers[0]) / 2
        y_shift = (y_centers[1] - y_centers[0]) / 2
        z_shift = (z_centers[1] - z_centers[0]) / 2
        x_centers = x_centers[:feature_map_size[1]] + x_shift # (feature_map_size[1], )
        y_centers = y_centers[:feature_map_size[0]] + y_shift # (feature_map_size[0], )
        z_centers = z_centers[:1] + z_shift  # (1, )

        # [feature_map_size[1], feature_map_size[0], 1, 2] * 4
        meshgrids = torch.meshgrid(x_centers, y_centers, z_centers, rotations)
        meshgrids = list(meshgrids)
        for i in range(len(meshgrids)):
            meshgrids[i] = meshgrids[i][..., None] # [feature_map_size[1], feature_map_size[0], 1, 2, 1]
        
        anchor_size = anchor_size[None, None, None, None, :]
        repeat_shape = [feature_map_size[1], feature_map_size[0], 1, len(rotations), 1]
        anchor_size = anchor_size.repeat(repeat_shape) # [feature_map_size[1], feature_map_size[0], 1, 2, 3]
        meshgrids.insert(3, anchor_size)
        anchors = torch.cat(meshgrids, dim=-1).permute(2, 1, 0, 3, 4).contiguous() # [1, feature_map_size[0], feature_map_size[1], 2, 7]
        return anchors.squeeze(0)


    def get_multi_anchors(self, feature_map_size):
        '''
        feature_map_size: (y_l, x_l)
        ranges: [[x1, y1, z1, x2, y2, z2], [x1, y1, z1, x2, y2, z2], [x1, y1, z1, x2, y2, z2]]
        sizes: [[w, l, h], [w, l, h], [w, l, h]]
        rotations: [0, 1.57]
        return: shape=(y_l, x_l, 3, 2, 7)
        '''
        device = feature_map_size.device
        ranges = torch.tensor(self.ranges, device=device)  
        sizes = torch.tensor(self.sizes, device=device) 
        rotations = torch.tensor(self.rotations, device=device)
        multi_anchors = []
        for i in range(len(ranges)):
            anchors = self.get_anchors(feature_map_size=feature_map_size, 
                                       anchor_range=ranges[i], 
                                       anchor_size=sizes[i], 
                                       rotations=rotations)
            multi_anchors.append(anchors[:, :, None, :, :])
        multi_anchors = torch.cat(multi_anchors, dim=2)

        return multi_anchors

            
class PillarLayer(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels):
        super().__init__()
        self.voxel_layer = Voxelization(voxel_size=voxel_size,
                                        point_cloud_range=point_cloud_range,
                                        max_num_points=max_num_points,
                                        max_voxels=max_voxels)

    @torch.no_grad()
    def forward(self, batched_pts):
        '''
        batched_pts: list[tensor], len(batched_pts) = bs
        return: 
               pillars: (p1 + p2 + ... + pb, num_points, c), 
               coors_batch: (p1 + p2 + ... + pb, 1 + 3), 
               num_points_per_pillar: (p1 + p2 + ... + pb, ), (b: batch size)
        '''
        pillars, coors, npoints_per_pillar = [], [], []
        for i, pts in enumerate(batched_pts):
            voxels_out, coors_out, num_points_per_voxel_out = self.voxel_layer(pts) 
            # voxels_out: (max_voxel, num_points, c), coors_out: (max_voxel, 3)
            # num_points_per_voxel_out: (max_voxel, )
            pillars.append(voxels_out)
            coors.append(coors_out.long())
            npoints_per_pillar.append(num_points_per_voxel_out)
        
        pillars = torch.cat(pillars, dim=0) # (p1 + p2 + ... + pb, num_points, c)
        npoints_per_pillar = torch.cat(npoints_per_pillar, dim=0) # (p1 + p2 + ... + pb, )
        coors_batch = []
        for i, cur_coors in enumerate(coors):
            coors_batch.append(F.pad(cur_coors, (1, 0), value=i))
        coors_batch = torch.cat(coors_batch, dim=0) # (p1 + p2 + ... + pb, 1 + 3)

        return pillars, coors_batch, npoints_per_pillar


class PillarEncoder(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, in_channel, out_channel):
        super().__init__()
        self.out_channel = out_channel
        self.vx, self.vy = voxel_size[0], voxel_size[1]
        self.x_offset = voxel_size[0] / 2 + point_cloud_range[0]
        self.y_offset = voxel_size[1] / 2 + point_cloud_range[1]
        self.x_l = math.ceil((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
        self.y_l = math.ceil((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])

        self.conv = nn.Conv1d(in_channel, out_channel, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_channel, eps=1e-3, momentum=0.01)

    @torch.no_grad()
    def forward(self, pillars, coors_batch, npoints_per_pillar):
        '''
        pillars: (p1 + p2 + ... + pb, num_points, c), c = 4
        coors_batch: (p1 + p2 + ... + pb, 1 + 3)
        npoints_per_pillar: (p1 + p2 + ... + pb, )
        return:  (bs, out_channel, y_l, x_l)
        '''
        device = pillars.device
        # 1. calculate offset to the points center (in each pillar)
        offset_pt_center = pillars[:, :, :3] - torch.sum(pillars[:, :, :3], dim=1, keepdim=True) / npoints_per_pillar[:, None, None] # (p1 + p2 + ... + pb, num_points, 3)

        # 2. calculate offset to the pillar center
        x_offset_pi_center = pillars[:, :, :1] - (coors_batch[:, None, 1:2] * self.vx + self.x_offset) # (p1 + p2 + ... + pb, num_points, 1)
        y_offset_pi_center = pillars[:, :, 1:2] - (coors_batch[:, None, 2:3] * self.vy + self.y_offset) # (p1 + p2 + ... + pb, num_points, 1)

        # 3. encoder
        features = torch.cat([pillars, offset_pt_center, x_offset_pi_center, y_offset_pi_center], dim=-1) # (p1 + p2 + ... + pb, num_points, 9)
        features[:, :, 0:1] = x_offset_pi_center # tmp
        features[:, :, 1:2] = y_offset_pi_center # tmp
        # In consitent with mmdet3d. 
        # The reason can be referenced to https://github.com/open-mmlab/mmdetection3d/issues/1150

        # 4. find mask for (0, 0, 0) and update the encoded features
        # a very beautiful implementation
        voxel_ids = torch.arange(0, pillars.size(1)).to(device) # (num_points, )
        mask = voxel_ids[:, None] < npoints_per_pillar[None, :] # (num_points, p1 + p2 + ... + pb)
        mask = mask.permute(1, 0).contiguous()  # (p1 + p2 + ... + pb, num_points)
        features *= mask[:, :, None]

        # 5. embedding
        features = features.permute(0, 2, 1).contiguous() # (p1 + p2 + ... + pb, 9, num_points)
        features = F.relu(self.bn(self.conv(features)))  # (p1 + p2 + ... + pb, out_channels, num_points)
        pooling_features = torch.max(features, dim=-1)[0] # (p1 + p2 + ... + pb, out_channels)

        # 6. pillar scatter
        batched_canvas = []
        bs = coors_batch[-1, 0] + 1
        for i in range(bs):
            cur_coors_idx = coors_batch[:, 0] == i
            cur_coors = coors_batch[cur_coors_idx, :]
            cur_features = pooling_features[cur_coors_idx]

            canvas = torch.zeros((self.x_l, self.y_l, self.out_channel), dtype=torch.float32, device=device)
            canvas[cur_coors[:, 1], cur_coors[:, 2]] = cur_features
            canvas = canvas.permute(2, 1, 0).contiguous()
            batched_canvas.append(canvas)
        batched_canvas = torch.stack(batched_canvas, dim=0) # (bs, in_channel, self.y_l, self.x_l)
        return batched_canvas


class Backbone(nn.Module):
    def __init__(self, in_channel, out_channels, layer_nums, layer_strides=[2, 2, 2]):
        super().__init__()
        assert len(out_channels) == len(layer_nums)
        assert len(out_channels) == len(layer_strides)
        
        self.multi_blocks = nn.ModuleList()
        for i in range(len(layer_strides)):
            blocks = []
            blocks.append(nn.Conv2d(in_channel, out_channels[i], 3, stride=layer_strides[i], bias=False, padding=1))
            blocks.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
            blocks.append(nn.ReLU(inplace=True))

            for _ in range(layer_nums[i]):
                blocks.append(nn.Conv2d(out_channels[i], out_channels[i], 3, bias=False, padding=1))
                blocks.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
                blocks.append(nn.ReLU(inplace=True))

            in_channel = out_channels[i]
            self.multi_blocks.append(nn.Sequential(*blocks))

        # in consitent with mmdet3d
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    @torch.no_grad()
    def forward(self, x):
        '''
        x: (b, c, y_l, x_l). Default: (6, 64, 496, 432)
        return: list[]. Default: [(6, 64, 248, 216), (6, 128, 124, 108), (6, 256, 62, 54)]
        '''
        outs = []
        for i in range(len(self.multi_blocks)):
            x = self.multi_blocks[i](x)
            outs.append(x)
        return outs


class Neck(nn.Module):
    def __init__(self, in_channels, upsample_strides, out_channels):
        super().__init__()
        assert len(in_channels) == len(upsample_strides)
        assert len(upsample_strides) == len(out_channels)

        self.decoder_blocks = nn.ModuleList()
        for i in range(len(in_channels)):
            decoder_block = []
            decoder_block.append(nn.ConvTranspose2d(in_channels[i], 
                                                    out_channels[i], 
                                                    upsample_strides[i], 
                                                    stride=upsample_strides[i],
                                                    bias=False))
            decoder_block.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
            decoder_block.append(nn.ReLU(inplace=True))

            self.decoder_blocks.append(nn.Sequential(*decoder_block))
        
        # in consitent with mmdet3d
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    @torch.no_grad()
    def forward(self, x):
        '''
        x: [(bs, 64, 248, 216), (bs, 128, 124, 108), (bs, 256, 62, 54)]
        return: (bs, 384, 248, 216)
        '''
        outs = []
        for i in range(len(self.decoder_blocks)):
            xi = self.decoder_blocks[i](x[i]) # (bs, 128, 248, 216)
            outs.append(xi)
        out = torch.cat(outs, dim=1)
        return out


class Head(nn.Module):
    def __init__(self, in_channel, n_anchors, n_classes):
        super().__init__()
        
        self.conv_cls = nn.Conv2d(in_channel, n_anchors*n_classes, 1)
        self.conv_reg = nn.Conv2d(in_channel, n_anchors*7, 1)
        self.conv_dir_cls = nn.Conv2d(in_channel, n_anchors*2, 1)

        # in consitent with mmdet3d
        conv_layer_id = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if conv_layer_id == 0:
                    prior_prob = 0.01
                    bias_init = float(-1 * math.log((1 - prior_prob) / prior_prob))
                    nn.init.constant_(m.bias, bias_init)
                else:
                    nn.init.constant_(m.bias, 0)
                conv_layer_id += 1

    @torch.no_grad()
    def forward(self, x):
        '''
        x: (bs, 384, 248, 216)
        return: 
              bbox_cls_pred: (bs, n_anchors*3, 248, 216) 
              bbox_pred: (bs, n_anchors*7, 248, 216)
              bbox_dir_cls_pred: (bs, n_anchors*2, 248, 216)
        '''
        bbox_cls_pred = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        bbox_dir_cls_pred = self.conv_dir_cls(x)
        return bbox_cls_pred, bbox_pred, bbox_dir_cls_pred


class PointPillars(nn.Module):
    def __init__(self,
                 nclasses=3, 
                 voxel_size=[0.32, 0.32, 6],
                 point_cloud_range=[-74.88, -74.88, -2, 74.88, 74.88, 4],
                 max_num_points=20,
                 max_voxels=(32000, 32000),
                 checkpoint=None,
                 target_device='cpu'):
        super().__init__()
        self.device = torch.device(target_device)
        self.nclasses = nclasses

        self.pillar_layer = PillarLayer(voxel_size=voxel_size, 
                                        point_cloud_range=point_cloud_range, 
                                        max_num_points=max_num_points, 
                                        max_voxels=max_voxels).to(self.device)
        pillar_channel = 16
        self.pillar_encoder = PillarEncoder(voxel_size=voxel_size, 
                                            point_cloud_range=point_cloud_range, 
                                            in_channel=pillar_channel,
                                            out_channel=64).to(self.device)
        self.backbone = Backbone(in_channel=64, 
                                 out_channels=[64, 128, 256], 
                                 layer_nums=[3, 5, 5],
                                 layer_strides=[1,2,2]).to(self.device)
        self.neck = Neck(in_channels=[64, 128, 256], 
                         upsample_strides=[1, 2, 4], 
                         out_channels=[128, 128, 128]).to(self.device)
        self.head = Head(in_channel=384, n_anchors=2*nclasses, 
                         n_classes=nclasses).to(self.device)
        
        # anchors
        ranges = [[-74.88, -74.88, -0.0345, 74.88, 74.88, -0.0345],
                    [-74.88, -74.88, 0, 74.88, 74.88, 0],
                    [-74.88, -74.88, -0.1188, 74.88, 74.88, -0.1188]]
        sizes = [[0.84, .91, 1.74], [.84, 1.81, 1.77], [2.08, 4.73, 1.77]]
        rotations=[0, 1.57]
        self.anchors_generator = Anchors(ranges=ranges, 
                                         sizes=sizes, 
                                         rotations=rotations,
                                         target_device=target_device)
        
        # val and test
        self.nms_pre = 4096
        self.nms_thr = 0.25
        self.score_thr = 0.1
        self.max_num = 500

        if checkpoint is not None:
            loaded_state = torch.load(checkpoint, map_location=self.device)
            self.load_state_dict(loaded_state["model_state_dict"])


    def anchors2bboxes(self, anchors, deltas):
        '''
        anchors: (M, 7),  (x, y, z, w, l, h, theta)
        deltas: (M, 7)
        return: (M, 7)
        '''
        da = torch.sqrt(anchors[:, 3] ** 2 + anchors[:, 4] ** 2)
        x = deltas[:, 0] * da + anchors[:, 0]
        y = deltas[:, 1] * da + anchors[:, 1]
        z = deltas[:, 2] * anchors[:, 5] + anchors[:, 2] + anchors[:, 5] / 2

        w = anchors[:, 3] * torch.exp(deltas[:, 3])
        l = anchors[:, 4] * torch.exp(deltas[:, 4])
        h = anchors[:, 5] * torch.exp(deltas[:, 5])

        z = z - h / 2

        theta = anchors[:, 6] + deltas[:, 6]
        
        bboxes = torch.stack([x, y, z, w, l, h, theta], dim=1)
        return bboxes

    def get_predicted_bboxes_single(self, bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchors):
        '''
        bbox_cls_pred: (n_anchors*3, 248, 216) 
        bbox_pred: (n_anchors*7, 248, 216)
        bbox_dir_cls_pred: (n_anchors*2, 248, 216)
        anchors: (y_l, x_l, 3, 2, 7)
        return: 
            bboxes: (k, 7)
            labels: (k, )
            scores: (k, ) 
        '''
        # modified from https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/core/bbox/structures/utils.py#L11
        def limit_period(val, offset=0.5, period=math.pi):
            """
            val: array or float
            offset: float
            period: float
            return: Value in the range of [-offset * period, (1-offset) * period]
            """
            limited_val = val - torch.floor(val / period + offset) * period
            return limited_val

        # 0. pre-process 
        bbox_cls_pred = bbox_cls_pred.permute(1, 2, 0).reshape(-1, self.nclasses)
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 7)
        bbox_dir_cls_pred = bbox_dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
        anchors = anchors.reshape(-1, 7)
        
        bbox_cls_pred = torch.sigmoid(bbox_cls_pred)
        bbox_dir_cls_pred = torch.max(bbox_dir_cls_pred, dim=1)[1]

        # 1. obtain self.nms_pre bboxes based on scores
        inds = bbox_cls_pred.max(1)[0].topk(self.nms_pre)[1]
        bbox_cls_pred = bbox_cls_pred[inds]
        bbox_pred = bbox_pred[inds]
        bbox_dir_cls_pred = bbox_dir_cls_pred[inds]
        anchors = anchors[inds]

        # 2. decode predicted offsets to bboxes
        bbox_pred = self.anchors2bboxes(anchors, bbox_pred)

        # 3. nms
        bbox_pred2d_xy = bbox_pred[:, [0, 1]]
        bbox_pred2d_lw = bbox_pred[:, [3, 4]]
        bbox_pred2d = torch.cat([bbox_pred2d_xy - bbox_pred2d_lw / 2,
                                 bbox_pred2d_xy + bbox_pred2d_lw / 2,
                                 bbox_pred[:, 6:]], dim=-1) # (n_anchors, 5)

        ret_bboxes, ret_labels, ret_scores = [], [], []
        for i in range(self.nclasses):
            # 3.1 filter bboxes with scores below self.score_thr
            cur_bbox_cls_pred = bbox_cls_pred[:, i]
            score_inds = cur_bbox_cls_pred > self.score_thr
            if score_inds.sum() == 0:
                continue

            cur_bbox_cls_pred = cur_bbox_cls_pred[score_inds]
            cur_bbox_pred2d = bbox_pred2d[score_inds]
            cur_bbox_pred = bbox_pred[score_inds]
            cur_bbox_dir_cls_pred = bbox_dir_cls_pred[score_inds]
            
            # 3.2 nms core
            keep_inds = nms_cuda(boxes=cur_bbox_pred2d, 
                                 scores=cur_bbox_cls_pred, 
                                 thresh=self.nms_thr, 
                                 pre_maxsize=None, 
                                 post_max_size=None)

            cur_bbox_cls_pred = cur_bbox_cls_pred[keep_inds]
            cur_bbox_pred = cur_bbox_pred[keep_inds]
            cur_bbox_dir_cls_pred = cur_bbox_dir_cls_pred[keep_inds]
            cur_bbox_pred[:, -1] = limit_period(cur_bbox_pred[:, -1], 1, math.pi).to(cur_bbox_pred) # [-pi, 0]
            cur_bbox_pred[:, -1] += (1 - cur_bbox_dir_cls_pred) * math.pi

            ret_bboxes.append(cur_bbox_pred)
            ret_labels.append(torch.zeros_like(cur_bbox_pred[:, 0], dtype=torch.long) + i)
            ret_scores.append(cur_bbox_cls_pred)

        # 4. filter some bboxes if bboxes number is above self.max_num
        if len(ret_bboxes) == 0:
            return {
            'lidar_bboxes': torch.empty((0, 7)).detach().cpu(),
            'labels': torch.empty(0).detach().cpu(),
            'scores': torch.empty(0).detach().cpu()
            }
        ret_bboxes = torch.cat(ret_bboxes, 0)
        ret_labels = torch.cat(ret_labels, 0)
        ret_scores = torch.cat(ret_scores, 0)
        if ret_bboxes.size(0) > self.max_num:
            final_inds = ret_scores.topk(self.max_num)[1]
            ret_bboxes = ret_bboxes[final_inds]
            ret_labels = ret_labels[final_inds]
            ret_scores = ret_scores[final_inds]
        result = {
            'lidar_bboxes': ret_bboxes.detach().cpu(),
            'labels': ret_labels.detach().cpu(),
            'scores': ret_scores.detach().cpu()
        }
        return result


    def get_predicted_bboxes(self, bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, batched_anchors):
        '''
        bbox_cls_pred: (bs, n_anchors*3, 248, 216) 
        bbox_pred: (bs, n_anchors*7, 248, 216)
        bbox_dir_cls_pred: (bs, n_anchors*2, 248, 216)
        batched_anchors: (bs, y_l, x_l, 3, 2, 7)
        return: 
            bboxes: [(k1, 7), (k2, 7), ... ]
            labels: [(k1, ), (k2, ), ... ]
            scores: [(k1, ), (k2, ), ... ] 
        '''
        results = []
        bs = bbox_cls_pred.size(0)
        for i in range(bs):
            result = self.get_predicted_bboxes_single(bbox_cls_pred=bbox_cls_pred[i],
                                                      bbox_pred=bbox_pred[i], 
                                                      bbox_dir_cls_pred=bbox_dir_cls_pred[i], 
                                                      anchors=batched_anchors[i])
            results.append(result)
        return results


    @torch.no_grad()
    def forward(self, batched_pts):
        batch_size = len(batched_pts)
        # batched_pts: list[tensor] -> pillars: (p1 + p2 + ... + pb, num_points, c), 
        #                              coors_batch: (p1 + p2 + ... + pb, 1 + 3), 
        #                              num_points_per_pillar: (p1 + p2 + ... + pb, ), (b: batch size)
        pillars, coors_batch, npoints_per_pillar = self.pillar_layer(batched_pts)

        # pillars: (p1 + p2 + ... + pb, num_points, c), c = 4
        # coors_batch: (p1 + p2 + ... + pb, 1 + 3)
        # npoints_per_pillar: (p1 + p2 + ... + pb, )
        #                     -> pillar_features: (bs, out_channel, y_l, x_l)
        pillar_features = self.pillar_encoder(pillars, coors_batch, npoints_per_pillar)

        # xs:  [(bs, 64, 248, 216), (bs, 128, 124, 108), (bs, 256, 62, 54)]
        xs = self.backbone(pillar_features)

        # x: (bs, 384, 248, 216)
        x = self.neck(xs)

        # bbox_cls_pred: (bs, n_anchors*3, 248, 216) 
        # bbox_pred: (bs, n_anchors*7, 248, 216)
        # bbox_dir_cls_pred: (bs, n_anchors*2, 248, 216)
        bbox_cls_pred, bbox_pred, bbox_dir_cls_pred = self.head(x)

        # anchors
        feature_map_size = torch.tensor(list(bbox_cls_pred.size()[-2:]), device=self.device)
        anchors = self.anchors_generator.get_multi_anchors(feature_map_size)
        batched_anchors = [anchors for _ in range(batch_size)]

        results = self.get_predicted_bboxes(bbox_cls_pred=bbox_cls_pred, 
                                            bbox_pred=bbox_pred, 
                                            bbox_dir_cls_pred=bbox_dir_cls_pred, 
                                            batched_anchors=batched_anchors)
        return results
        
