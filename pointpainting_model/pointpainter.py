import torch

class PointPainter(nn.Module):
    def __init__(self, cam_sync=False, target_device='cpu'):
        super().__init__()
        self.seg_net_index = 0
        self.cam_sync = cam_sync
        self.device = torch.device(target_device)

    def get_score(self, x):
        sf = torch.nn.Softmax(dim=2)
        output_permute = x.permute(1,2,0)
        output_permute = sf(output_permute)
        output_reassign = torch.zeros(output_permute.size(0),output_permute.size(1), 6).to(device=x.device)
        output_reassign[:,:,0] = torch.sum(output_permute[:,:,:11], dim=2) # background
        output_reassign[:,:,1] = output_permute[:,:,18] #bicycle
        output_reassign[:,:,2] = torch.sum(output_permute[:,:,[13, 14, 15, 16]], dim=2) # vehicles
        output_reassign[:,:,3] = output_permute[:,:,11] #person
        output_reassign[:,:,4] = output_permute[:,:,12] #rider
        output_reassign[:,:,5] = output_permute[:,:,17] # motorcycle

        return output_reassign
    
    def cam_to_lidar(self, pointcloud, projection_mats, camera_num):
        """
        Takes in lidar in velo coords, returns lidar points in camera coords

        :param pointcloud: (n_points, 4) array (x,y,z,r) in velodyne coordinates
        :return lidar_cam_coords: (n_points, 4) array (x,y,z,r) in camera coordinates
        """

        lidar_velo_coords = pointcloud.clone()
        reflectances = lidar_velo_coords[:, -1].clone() #copy reflectances column
        lidar_velo_coords[:, -1] = 1 # for multiplying with homogeneous matrix
        lidar_cam_coords = projection_mats['Tr_velo_to_cam_' + str(camera_num)].matmul(lidar_velo_coords.transpose(0, 1))
        lidar_cam_coords = lidar_cam_coords.transpose(0, 1)
        lidar_cam_coords[:, -1] = reflectances
        
        return lidar_cam_coords

    def project_points_mask(self, lidar_cam_points, projection_mats, class_scores, camera_num):
        points_projected_on_mask = projection_mats['P' + str(camera_num)].matmul(projection_mats['R0_rect'].matmul(lidar_cam_points.transpose(0, 1)))
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
        lidar_cam_coords = self.cam_to_lidar(lidar_raw[:,:4], projection_mats, 0)

        # right
        lidar_cam_coords[:, -1] = 1 #homogenous coords for projection
        # TODO: change projection_mats['P2'] and projection_mats['R0_rect'] to be?
        points_projected_on_mask_0, true_where_point_on_img_0 = self.project_points_mask(lidar_cam_coords, projection_mats, class_scores, 0)
        
        # left
        lidar_cam_coords = self.cam_to_lidar(lidar_raw[:,:4], projection_mats, 1)
        lidar_cam_coords[:, -1] = 1 #homogenous coords for projection
        # TODO: change projection_mats['P2'] and projection_mats['R0_rect'] to be?
        points_projected_on_mask_1, true_where_point_on_img_1 = self.project_points_mask(lidar_cam_coords, projection_mats, class_scores, 1)
        
        lidar_cam_coords = self.cam_to_lidar(lidar_raw[:,:4], projection_mats, 2)
        lidar_cam_coords[:, -1] = 1
        points_projected_on_mask_2, true_where_point_on_img_2 = self.project_points_mask(lidar_cam_coords, projection_mats, class_scores, 2)
        
        lidar_cam_coords = self.cam_to_lidar(lidar_raw[:,:4], projection_mats, 3)
        lidar_cam_coords[:, -1] = 1
        points_projected_on_mask_3, true_where_point_on_img_3 = self.project_points_mask(lidar_cam_coords, projection_mats, class_scores, 3)
        
        lidar_cam_coords = self.cam_to_lidar(lidar_raw[:,:4], projection_mats, 4)
        lidar_cam_coords[:, -1] = 1
        points_projected_on_mask_4, true_where_point_on_img_4 = self.project_points_mask(lidar_cam_coords, projection_mats, class_scores, 4)

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
    def forward(self, segmentation_results, lidar_raw, projection_mats):
        return self.augment_lidar_class_scores_both(self.get_score(segmentation_results), lidar_raw, projection_mats)