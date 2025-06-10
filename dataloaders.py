import open3d as o3d
import os
import numpy as np
import torch
import json
import cv2
from torch.utils.data import Dataset

from sklearn.neighbors import KDTree

import scipy
import random

class PCLoader(Dataset):
    def __init__(self, num_points=32768, nn=20, partition='train', dataset_index=0, data_path="/workspace/bop/icbin/train_pbr/", image_ext=".jpg", camera_param_path="/workspace/bop/icbin/train_pbr/000000/scene_camera.json", im_w=640, im_h=480, train_cad_list = ["/workspace/bop/icbin/models/obj_000001.ply", "/workspace/bop/icbin/models/obj_000002.ply"], normal="zaxis", veclen=None):

        self.num_points = num_points
        self.nn = nn
        self.partition = partition

        folder_path = os.path.expanduser('~')

        CAMERA_PARAM = folder_path + camera_param_path

        camera_param = json.load(open(CAMERA_PARAM))

        self.cameraMat = camera_param['0']['cam_K']
        self.cameraMat = np.reshape(self.cameraMat, (3,3))
        self.depth_scale = camera_param['0']['depth_scale']
        
        self.image_ext = image_ext
        self.data_path = folder_path + data_path

        self.train_list = []

        self.im_w, self.im_h = im_w, im_h 

        for train_index, cad_string in enumerate(train_cad_list):
            train_index += 1
            
            obj_cad = o3d.io.read_triangle_mesh(folder_path + cad_string)
            obj_pc = obj_cad.sample_points_poisson_disk(2048)

            obj_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30), False)

            object_xyz = np.asarray(obj_pc.points)

            o3d_bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(obj_pc.points)
            bb83d = np.asarray(o3d_bbox.get_box_points())
            
            if normal == "zaxis":
                center_point = np.array([0, 0, 0])
                top_point = np.array([0, 0, bb83d[6,2]]) 
            elif normal == "xaxis":
                top_point = center_point + np.array([bb83d[1,0], 0, 0]) 

            self.train_list.append( { "obj": object_xyz, "idx": train_index, "gtp": None, "bb83d": bb83d, "center": center_point, "top": top_point} )

        self.dataset_index = dataset_index

        gt_file = self.data_path + str(dataset_index).zfill(6) + "/scene_gt.json"
        gty = json.load(open(gt_file))
        try:
            gt_file = self.data_path + str(dataset_index).zfill(6) + "/scene_gt_info.json"
            gty_info = json.load(open(gt_file))
            self.gtreadvis(gty, gty_info)
        except:
            self.gtreadvis(gty)

        print( "len(gty)", len(gty) )
        if veclen is None:
            self.file_names = range(0,len(gty))
        else:
            self.file_names = range(0,veclen)

        print( "Success!", dataset_index )

    def gtreadvis(self, gty, gty_info ):
        for training_instance in self.train_list:
            gt_poses = {}
            for key in gty.keys():
                tempPoseList = []
                for i in range( len( gty[key] )):
                    point = gty[key][i]['cam_t_m2c']
                    rot = np.array( gty[key][i]['cam_R_m2c'] ).reshape( (3,3) )
                    if( gty[key][i]['obj_id'] == int(training_instance['idx'])):
                        tempPoseList.append( (point, rot, gty_info[key][i]["visib_fract"]) )   
                gt_poses[key] = tempPoseList
            training_instance['gtp'] = gt_poses
    
    def gtreadvis(self, gty):
        for training_instance in self.train_list:
            gt_poses = {}
            for key in gty.keys():
                tempPoseList = []
                for i in range( len( gty[key] )):
                    point = gty[key][i]['cam_t_m2c']
                    rot = np.array( gty[key][i]['cam_R_m2c'] ).reshape( (3,3) )
                    if( gty[key][i]['obj_id'] == int(training_instance['idx'])):
                        tempPoseList.append( (point, rot, 1) )   
                gt_poses[key] = tempPoseList
            training_instance['gtp'] = gt_poses
    
    def __getitem__(self, item):
        dataset_index = self.dataset_index
        current_scene_index = item
       
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(self.im_w, self.im_h, self.cameraMat[0,0], self.cameraMat[1,1], self.cameraMat[0,2], self.cameraMat[1,2])

        depth = cv2.imread(self.data_path + str(dataset_index).zfill(6) + "/depth/" + str(current_scene_index).zfill(6) + ".png", cv2.IMREAD_ANYDEPTH)

        depth = np.array(depth, np.float64)

        depth_trunc = 2500.0

        depth = depth * self.depth_scale

        return_depth_scale = 10
       
        pc_scaled = False

        gaussian_noise = 0

        if self.partition == "train":
            depth, gaussian_noise = add_depth_image_noise(depth=depth)
            if np.random.random() < 0.05:
                depth, depth_trunc = random_scale(depth, depth_trunc)
                pc_scaled = True

        depth = np.array(depth*10, np.uint16)
        depth_raw = o3d.geometry.Image(depth)

        scene_pointcloud = o3d.geometry.PointCloud.create_from_depth_image(depth_raw, intrinsic, depth_scale=return_depth_scale, depth_trunc=depth_trunc, project_valid_depth_only=True)

        if np.asarray(scene_pointcloud.points).shape[0] < self.num_points/16:
            depth = cv2.imread(self.data_path + str(dataset_index).zfill(6) + "/depth/" + str(current_scene_index).zfill(6) + ".png", cv2.IMREAD_ANYDEPTH)
            depth = np.array(depth, np.float64)
            depth = depth * self.depth_scale
            depth = np.array(depth*10, np.uint16)
            depth_raw = o3d.geometry.Image(depth)

            depth_trunc = 5000.0 
            scene_pointcloud = o3d.geometry.PointCloud.create_from_depth_image(depth_raw, intrinsic, depth_scale=10, depth_trunc=depth_trunc, project_valid_depth_only=True)
            scene_pointcloud = scene_pointcloud.voxel_down_sample(1)

        pointcloud = np.asarray(scene_pointcloud.points)
        index_list = np.arange(len(pointcloud))   
        while len(index_list) < self.num_points:
            index_list = np.array(list(index_list) + list(index_list))
        np.random.shuffle(index_list)
        pointcloud = pointcloud[ index_list, : ]

        pointcloud = pointcloud[ :self.num_points, : ]

        scene_pointcloud.points = o3d.utility.Vector3dVector(pointcloud) 

        scene_pointcloud.estimate_normals( o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=10), fast_normal_computation=True)
        scene_pointcloud.orient_normals_to_align_with_direction(orientation_reference=np.array([0.0, 0.0, -1.0]))

        pointcloud = np.concatenate([np.asarray(scene_pointcloud.points), np.asarray(scene_pointcloud.normals)], axis = 1)

        pointcloud = pointcloud.astype('float32')

        lastdim = 32
        
        neighbor_idx = []
        for neighbor_size, point_layer_size, k in [(self.num_points,self.num_points//4, self.nn+1),(self.num_points//4,self.num_points//8, self.nn+1),(self.num_points//8,self.num_points//16,self.nn+1),(self.num_points//16,self.num_points//32,self.nn+1),(self.num_points//32,self.num_points//32,self.nn+1) ]:
            kdtree = KDTree( pointcloud[:neighbor_size,:3])
            neighbors = kdtree.query( pointcloud[:point_layer_size,:3], k=k, return_distance=False, dualtree=False, sort_results=False)
            neighbors = neighbors[:,1:] # to avoid itself
            neighbor_idx.append(neighbors)
        neighbor_idx_val = np.concatenate(neighbor_idx, axis=0)

        
        pointcloud_input = pointcloud[:self.num_points//lastdim,:]

        cat = np.zeros( (self.num_points//lastdim), int )
        seg = np.zeros( (self.num_points//lastdim), int )
        obj_visibility = np.ones( (self.num_points//lastdim), np.double)        
        dist_to_center = np.zeros( (self.num_points//lastdim, 3), np.double)
        dist_to_top = np.zeros( (self.num_points//lastdim, 3), np.double)

        if pc_scaled:
            seg = cat[:self.num_points//lastdim]
            seg = torch.LongTensor(seg)

            cen = dist_to_center[:self.num_points//lastdim,:]
            top = dist_to_top[:self.num_points//lastdim,:]
            vis = obj_visibility[:self.num_points//lastdim]

            return pointcloud, seg, cen, top, neighbor_idx_val, vis

        for cat_idx, ti in enumerate(self.train_list):
            for seg_idx, gt_pose in enumerate(ti['gtp'][str(current_scene_index)]):
       
                newPL = np.matmul(ti['obj'], gt_pose[1].transpose() )
                newPL = newPL + gt_pose[0]
                


                treeLocal = KDTree(newPL, leaf_size=2)

                dist, ind = treeLocal.query(pointcloud_input[ : , :3 ], k=1)

                new_top = np.matmul(ti['top'], gt_pose[1].transpose() )
                new_top = new_top + gt_pose[0]


                new_center = np.matmul(ti['center'], gt_pose[1].transpose() )
                new_center = new_center + gt_pose[0]

                small_dist = dist < 5 + gaussian_noise
                small_dist = small_dist.flatten()

                if np.sum(small_dist) == 0:
                    continue
                    
                cat[ small_dist ] = cat_idx + 1

                dist_to_center[ small_dist, : ] = pointcloud_input[ small_dist , :3 ] - new_center
                dist_to_top[ small_dist, : ] = pointcloud_input[ small_dist , :3 ] - new_top

                obj_visibility[ small_dist ] = 1 + (1 - gt_pose[2])


        if 0: # visualize:
            point_list = pointcloud_input
            new_colors = []
            for i in range(len(cat)):
                if( cat[i] == 1 ):
                    new_colors.append([0,0,1])
                elif( cat[i] > 0 ):
                    new_colors.append([0,1,0])
                else:
                    new_colors.append([1,0,0])
            target = o3d.geometry.PointCloud()
            target.points = o3d.utility.Vector3dVector(np.array(point_list)[:,:3])
            target.normals = o3d.utility.Vector3dVector(np.array(point_list)[:,3:6])
            target.colors = o3d.utility.Vector3dVector(np.array(new_colors))
            o3d.visualization.draw_geometries([target]) 

        seg = cat[:self.num_points//lastdim]
        seg = torch.LongTensor(seg)

        cen = dist_to_center[:self.num_points//lastdim,:]
        top = dist_to_top[:self.num_points//lastdim,:]
        
        vis = obj_visibility[:self.num_points//lastdim]

    
        if self.partition == "train":
            if np.random.random() < 0.25:
                multiplier_vector = np.random.uniform(low=0.95, high=1.05, size=(3))
                pointcloud[:,:3] = pointcloud[:,:3] * multiplier_vector

        return pointcloud, seg, cen, top, neighbor_idx_val, vis

    def __len__(self):
        return len(self.file_names)

def add_depth_image_noise(depth):
    gaussian_noise = 0
    if np.random.random() < 0.2:
        src = np.array((depth/np.max(depth))*255,np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        edge = cv2.Canny(src, 60, 80)
        for _ in range(0,random.randint(0,3)):
            edge = cv2.dilate(edge,kernel)
        edge = 5000.0*(np.array(edge/255,np.float64))
        depth = depth + edge
    if np.random.random() < 0.20:
        number_of_iterations = 60  
        row,col = depth.shape
        circles = np.zeros((row,col),np.uint8)
        for _ in range(number_of_iterations):
            x = np.random.randint(0,col)
            y = np.random.randint(0,row)
            circles = cv2.circle(circles, (x,y), np.random.randint(5,10), 1, -1)
            depth = depth + (np.float64(circles)*50000.0)
    if np.random.random() < 0.20:
        number_of_iterations = 60
        row,col = depth.shape
        circles = np.zeros((row,col),np.uint8)
        for _ in range(number_of_iterations):
            x = np.random.randint(0,col)
            y = np.random.randint(0,row)
            circles = cv2.circle(circles, (x,y), np.random.randint(15,35), 1, -1)
            depth = depth + (np.float64(circles)*50000.0)
    if np.random.random() < 0.25:
        gaussian_noise = 5
        filter_size = int(np.random.randint(0,15)*2 + 3) #5
        depth = scipy.ndimage.median_filter(depth, size=(filter_size,filter_size))
    if np.random.random() < 0.25:
        row,col = depth.shape
        gaussian_noise = 1.0+np.random.random()*15
        gauss = np.random.randn(row,col)*gaussian_noise
        depth = depth + np.array(gauss,np.float32)         
    if np.random.random() < 0.10:
        col,row = depth.shape
        x = np.random.randint(int(col*0.25),int(col*0.75))
        y = np.random.randint(int(col*0.25),int(row*0.75))
        if np.random.random() < 0.5:
            depth[:y,:] = 50000
        else:
            depth[y:,:] = 50000
        if np.random.random() < 0.5:
            depth[:,:x] = 50000
        else:
            depth[:,x:] = 50000

    return depth, gaussian_noise

def random_scale(depth, depth_trunc):
    if np.random.random() < 0.5:
        return_depth_scale = np.random.uniform(2,5)
        depth_trunc = depth_trunc*10/return_depth_scale
    else:
        return_depth_scale = np.random.uniform(15,35)
        depth_trunc = depth_trunc*10/return_depth_scale
    return depth, depth_trunc