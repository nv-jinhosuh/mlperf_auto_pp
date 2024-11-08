import torch
import torch.nn as nn

# projection_tensor: ['Tr_velo_to_cam_0', 'Tr_velo_to_cam_1', 'Tr_velo_to_cam_2', 'Tr_velo_to_cam_3', 'Tr_velo_to_cam_4',
#                     'P0', 'P1', 'P2', 'P3', 'P4', 'R0_rect']

class PointPainter(nn.Module):
    def __init__(self, cam_sync=False, target_device='cpu'):
        super().__init__()

        self.device = torch.device(target_device)
        self.seg_net_index = 0
        self.cam_sync = cam_sync

    def get_score(self, x):
        sf = torch.nn.Softmax(dim=-1)
        output_permute = x.permute(0, 2, 3, 1)
        output_permute = sf(output_permute)
        output_reassign = torch.zeros(list(output_permute.shape[:-1]) + [6]).to(device=self.device)
        output_reassign[...,0] = torch.sum(output_permute[...,:11], dim=3) # background
        output_reassign[...,1] = output_permute[...,18] #bicycle
        output_reassign[...,2] = torch.sum(output_permute[...,[13, 14, 15, 16]], dim=3) # vehicles
        output_reassign[...,3] = output_permute[...,11] #person
        output_reassign[...,4] = output_permute[...,12] #rider
        output_reassign[...,5] = output_permute[...,17] # motorcycle

        return output_reassign
    
    def cam_to_lidar(self, pointcloud, tr_velo_to_cam_mat):
        """
        Takes in lidar in velo coords, returns lidar points in camera coords

        :param pointcloud: (n_points, 4) array (x,y,z,r) in velodyne coordinates
        :return lidar_cam_coords: (n_points, 4) array (x,y,z,r) in camera coordinates
        """

        lidar_velo_coords = pointcloud.clone()
        reflectances = lidar_velo_coords[:, -1].clone() #copy reflectances column
        lidar_velo_coords[:, -1] = 1 # for multiplying with homogeneous matrix
        lidar_cam_coords = tr_velo_to_cam_mat.matmul(lidar_velo_coords.transpose(0, 1))
        lidar_cam_coords = lidar_cam_coords.transpose(0, 1)
        lidar_cam_coords[:, -1] = reflectances
        
        return lidar_cam_coords

    def project_points_mask(self, lidar_cam_points, P_mat, R0_rect_mat, class_scores, camera_num):
        points_projected_on_mask = P_mat.matmul(R0_rect_mat.matmul(lidar_cam_points.transpose(0, 1)))
        points_projected_on_mask = points_projected_on_mask.transpose(0, 1)
        points_projected_on_mask = points_projected_on_mask/(points_projected_on_mask[:,2].reshape(-1,1))

        true_where_x_on_img = (0 < points_projected_on_mask[:, 0]) & (points_projected_on_mask[:, 0] < class_scores[camera_num].shape[1]) #x in img coords is cols of img
        true_where_y_on_img = (0 < points_projected_on_mask[:, 1]) & (points_projected_on_mask[:, 1] < class_scores[camera_num].shape[0])
        true_where_point_on_img = true_where_x_on_img & true_where_y_on_img & (lidar_cam_points[:, 2] > 0)

        points_projected_on_mask = points_projected_on_mask[true_where_point_on_img] # filter out points that don't project to image
        points_projected_on_mask = torch.floor(points_projected_on_mask).int() # using floor so you don't end up indexing num_rows+1th row or col
        points_projected_on_mask = points_projected_on_mask[:, :2] #drops homogenous coord 1 from every point, giving (N_pts, 2) int array
        return (points_projected_on_mask, true_where_point_on_img)

    def augment_lidar_class_scores_both(self, class_scores, lidar_raw, projection_mats):
        """
        Projects lidar points onto segmentation map, appends class score each point projects onto.
        """
        #lidar_cam_coords = self.cam_to_lidar(lidar_raw, projection_mats)
        # TODO: Project lidar points onto left and right segmentation maps. How to use projection_mats? 
        ################################

        lidar_cam_coords = self.cam_to_lidar(lidar_raw[:,:4], projection_mats[0])

        # right
        lidar_cam_coords[:, -1] = 1 #homogenous coords for projection
        # TODO: change projection_mats['P2'] and projection_mats['R0_rect'] to be?
        points_projected_on_mask_0, true_where_point_on_img_0 = self.project_points_mask(lidar_cam_coords, projection_mats[5], projection_mats[10], class_scores, 0)
        
        # left
        lidar_cam_coords = self.cam_to_lidar(lidar_raw[:,:4], projection_mats[1])
        lidar_cam_coords[:, -1] = 1 #homogenous coords for projection
        # TODO: change projection_mats['P2'] and projection_mats['R0_rect'] to be?
        points_projected_on_mask_1, true_where_point_on_img_1 = self.project_points_mask(lidar_cam_coords, projection_mats[6], projection_mats[10], class_scores, 1)
        
        lidar_cam_coords = self.cam_to_lidar(lidar_raw[:,:4], projection_mats[2])
        lidar_cam_coords[:, -1] = 1
        points_projected_on_mask_2, true_where_point_on_img_2 = self.project_points_mask(lidar_cam_coords, projection_mats[7], projection_mats[10], class_scores, 2)
        
        lidar_cam_coords = self.cam_to_lidar(lidar_raw[:,:4], projection_mats[3])
        lidar_cam_coords[:, -1] = 1
        points_projected_on_mask_3, true_where_point_on_img_3 = self.project_points_mask(lidar_cam_coords, projection_mats[8], projection_mats[10], class_scores, 3)
        
        lidar_cam_coords = self.cam_to_lidar(lidar_raw[:,:4], projection_mats[4])
        lidar_cam_coords[:, -1] = 1
        points_projected_on_mask_4, true_where_point_on_img_4 = self.project_points_mask(lidar_cam_coords, projection_mats[9], projection_mats[10], class_scores, 4)

        true_where_point_on_both_0_1 = true_where_point_on_img_0 & true_where_point_on_img_1
        true_where_point_on_both_0_2 = true_where_point_on_img_0 & true_where_point_on_img_2
        true_where_point_on_both_1_3 = true_where_point_on_img_1 & true_where_point_on_img_3
        true_where_point_on_both_2_4 = true_where_point_on_img_2 & true_where_point_on_img_4
        true_where_point_on_img = true_where_point_on_img_1 | true_where_point_on_img_0 | true_where_point_on_img_2 | true_where_point_on_img_3 | true_where_point_on_img_4

        point_scores_0 = class_scores[0][points_projected_on_mask_0[:, 1], points_projected_on_mask_0[:, 0]].reshape(-1, class_scores[0].shape[2])
        point_scores_1 = class_scores[1][points_projected_on_mask_1[:, 1], points_projected_on_mask_1[:, 0]].reshape(-1, class_scores[1].shape[2])
        point_scores_2 = class_scores[2][points_projected_on_mask_2[:, 1], points_projected_on_mask_2[:, 0]].reshape(-1, class_scores[2].shape[2])
        point_scores_3 = class_scores[3][points_projected_on_mask_3[:, 1], points_projected_on_mask_3[:, 0]].reshape(-1, class_scores[3].shape[2])
        point_scores_4 = class_scores[4][points_projected_on_mask_4[:, 1], points_projected_on_mask_4[:, 0]].reshape(-1, class_scores[4].shape[2])

        augmented_lidar = torch.cat((lidar_raw[:,:5], torch.zeros((lidar_raw.shape[0], class_scores[1].shape[2])).to(device=lidar_raw.device)), axis=1)
        augmented_lidar[true_where_point_on_img_0, -class_scores[0].shape[2]:] += point_scores_0
        augmented_lidar[true_where_point_on_img_1, -class_scores[1].shape[2]:] += point_scores_1
        augmented_lidar[true_where_point_on_img_2, -class_scores[2].shape[2]:] += point_scores_2
        augmented_lidar[true_where_point_on_img_3, -class_scores[3].shape[2]:] += point_scores_3
        augmented_lidar[true_where_point_on_img_4, -class_scores[4].shape[2]:] += point_scores_4
        augmented_lidar[true_where_point_on_both_0_1, -class_scores[0].shape[2]:] = 0.5 * augmented_lidar[true_where_point_on_both_0_1, -class_scores[0].shape[2]:]
        augmented_lidar[true_where_point_on_both_0_2, -class_scores[0].shape[2]:] = 0.5 * augmented_lidar[true_where_point_on_both_0_2, -class_scores[0].shape[2]:]
        augmented_lidar[true_where_point_on_both_1_3, -class_scores[1].shape[2]:] = 0.5 * augmented_lidar[true_where_point_on_both_1_3, -class_scores[1].shape[2]:]
        augmented_lidar[true_where_point_on_both_2_4, -class_scores[2].shape[2]:] = 0.5 * augmented_lidar[true_where_point_on_both_2_4, -class_scores[2].shape[2]:]
        if self.cam_sync:
            augmented_lidar = augmented_lidar[true_where_point_on_img]

        return augmented_lidar

    @torch.no_grad()
    def forward(self, segmentation_results, lidar_raw, proj_mtx):
        return self.augment_lidar_class_scores_both(self.get_score(segmentation_results), lidar_raw, proj_mtx)
