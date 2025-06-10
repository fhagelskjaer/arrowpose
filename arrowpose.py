from enum import Enum


import open3d as o3d
import numpy as np

from distinctipy import distinctipy
from sklearn.neighbors import KDTree
from scipy.spatial.transform import Rotation

import vis

class ICPTYPE(Enum):
    ONLY_POINTS = 0
    NONE = 1
    ICP = 2
    ICPROT = 3
    CLASSIC = 4

class ArrowPose:
    def __init__(self, cluster_radius=10, min_num_point_in_cluster=3, source=None, compute_transform_matrix=None):
        self.cluster_radius = cluster_radius
        self.min_num_point_in_cluster = min_num_point_in_cluster
        self.source = source
        self.compute_transform_matrix = compute_transform_matrix
        

    def compute_poses(self, search_seg_idx, seg_pred, cen_pred, top_pred, pointcloud_o3d, resized_pointcloud, refinement=None, visu=False, fig=False):

        seg_pred = seg_pred.permute(0, 2, 1).contiguous()
        cen_pred = cen_pred.permute(0, 2, 1).contiguous()
        top_pred = top_pred.permute(0, 2, 1).contiguous()

        cen_pred_np = cen_pred.view(-1,3).detach().cpu().numpy()
        top_pred_np = top_pred.view(-1,3).detach().cpu().numpy()

        # seg_pred_np = seg_pred.view(-1,3).detach().cpu().numpy()
        # seg_obj = seg_pred_np[:,2] > 0
        
        # import pdb; pdb.set_trace();

        seg_pred_max = seg_pred.max(dim=2)[1]
        seg_pred_max_np = seg_pred_max.view(-1,1).squeeze().detach().cpu().numpy()

        seg_obj = seg_pred_max_np == search_seg_idx
        # seg_obj = seg_pred_max_np != 1 # TODO

        cen_pred_np_seg = cen_pred_np[seg_obj,:]
        top_pred_np_seg = top_pred_np[seg_obj,:]
        segmented_points = resized_pointcloud[seg_obj,:]

        if visu:
            vis.visualize_prediction(pointcloud_o3d, segmented_points, cen_pred_np_seg, top_pred_np_seg, fig)

        center_points = segmented_points-np.array(cen_pred_np_seg)

        kdtree = KDTree( center_points )
        neighbors = kdtree.query_radius(center_points, r=self.cluster_radius)
        
        unique_labels = non_maximum_suppresion_of_nearest_neighbors(neighbors, self.min_num_point_in_cluster)

        if len(unique_labels) == 0:
            return []

        detected_center_points = center_points[ np.array(unique_labels) ]

        if visu:
            input_colors = [(1, 1, 1), [0,0,0]]
            colors_float = distinctipy.get_colors(len(unique_labels), input_colors)
            vis.visualize_segmentation(pointcloud_o3d, neighbors, unique_labels, center_points, segmented_points, colors_float, fig)

        if refinement == None:
            return detected_center_points

        # centers_scene_point = X[ np.array(unique_labels) ]
        # top_scene_point = (segmented_points-np.array(top_pred_np_seg))[ np.array(unique_labels) ]
        
        transform_list = []
        
        for label_index, label in enumerate(unique_labels): 
            
            # Top point of center
            # normal = top_scene_point[label_index] - detected_center_points[label_index]
            # point = detected_center_points[label_index]
            # transform = compute_transform_matrix(normal, point)
                    
            mean_center_point = np.mean( (segmented_points-np.array(cen_pred_np_seg))[neighbors[label]], axis=0)

            # Mean of top point center
            # normal_mean = mean_center_point - np.mean( (segmented_points-np.array(top_pred_np_seg))[ neighbors[label] ], axis=0) 

            # Searching for best top point center
            normal_mean = average_of_best_top_point(label, segmented_points, top_pred_np_seg, neighbors, mean_center_point, self.cluster_radius)

            transform = self.compute_transform_matrix(normal_mean, mean_center_point)

            transform, score = refinement(transform, segmented_points, neighbors, label, self.source, pointcloud_o3d)

            # print( "Center", len(neighbors[u_l]), len(top_neighbors[arg_len_top_array[0]]) )

            if score > 0:
                transform_list.append({"t": transform, "s": score})
            # transform_list.append({"t": transform, "s": len(neighbors[u_l])*score})
            
        if visu:
            vis.visualize_arrow(pointcloud_o3d, transform_list, colors_float, fig)
            vis.visualize_pose_estimation(pointcloud_o3d, self.source, transform_list, colors_float, fig)
        
        return transform_list
    

def average_of_best_top_point(unique_label, segmented_points, top_pred_np_seg, neighbors, center_point, cluster_radius):
    possible_top_points = (segmented_points-np.array(top_pred_np_seg))[ neighbors[unique_label] ]
    
    top_kdtree = KDTree( possible_top_points )
    top_neighbors = top_kdtree.query_radius(possible_top_points, r=cluster_radius)
    top_len_array = [len(c) for c in top_neighbors]

    arg_len_top_array = np.argsort( -np.array(top_len_array) )
    
    normal_mean = np.mean( possible_top_points[top_neighbors[arg_len_top_array[0]]], axis=0) - center_point

    return normal_mean

def non_maximum_suppresion_of_nearest_neighbors(neighbors, min_num_point_in_cluster):
    neighbor_list_len = []    
    for neighbor_list in neighbors:
        neighbor_list_len.append(len(neighbor_list))
    
    neighbor_list_len = np.array(neighbor_list_len)
    index_sorted_neighbor_list_len = np.argsort(-neighbor_list_len)
    
    unique_labels = []
    point_already_part_of_unique_label = set()
    
    for index_of_sorted_neighbors in index_sorted_neighbor_list_len:
    
        if neighbor_list_len[index_of_sorted_neighbors] < min_num_point_in_cluster:
            continue

        if (not(index_of_sorted_neighbors in point_already_part_of_unique_label)):
            unique_labels.append(index_of_sorted_neighbors)

        for neighbor_index in neighbors[index_of_sorted_neighbors]: 
            point_already_part_of_unique_label.add(neighbor_index)

    return unique_labels

class TransformMatrix:
    def __init__(self, obj_center=np.eye(4)):
        self.center_transform = obj_center

    def compute_transform_matrix_x(self, normal, point):
        """Compute a transformation matrix from a normal vector and a point."""
        # Normalize the input normal vector
        normal = normalize(normal)
        # Create an arbitrary vector that is not parallel to the normal
        if abs(normal[1]) < 1e-6 and abs(normal[2]) < 1e-6:
            tangent = np.array([0, 0, 1])  # Handle the case where the normal is along the x-axis
        else:
            tangent = np.array([1, 0, 0])
        z_axis = normalize(np.cross(tangent, normal))
        # Compute the second perpendicular vector (y-axis)
        y_axis = np.cross(normal, z_axis)
        # Construct the 4x4 transformation matrix
        transform_matrix = np.eye(4)
        transform_matrix[:3, 0] = normal
        transform_matrix[:3, 1] = y_axis
        transform_matrix[:3, 2] = z_axis
        transform_matrix[:3, 3] = point
        transform_matrix = transform_matrix @ self.center_transform
        return transform_matrix

    def compute_transform_matrix_z(self, normal, point):
        """Compute a transformation matrix from a normal vector and a point."""
        # Normalize the input normal vector
        normal = normalize(normal)

        # Create an arbitrary vector that is not parallel to the normal
        if abs(normal[0]) < 1e-6 and abs(normal[1]) < 1e-6:
            tangent = np.array([1, 0, 0])  # Handle the case where the normal is along the z-axis
        else:
            tangent = np.array([0, 0, 1])
        
        # Compute the first perpendicular vector (x-axis)
        x_axis = normalize(np.cross(tangent, normal))
        
        # Compute the second perpendicular vector (y-axis)
        y_axis = np.cross(normal, x_axis)

        # Construct the 4x4 transformation matrix
        transform_matrix = np.eye(4)
        transform_matrix[:3, 0] = x_axis
        transform_matrix[:3, 1] = y_axis
        transform_matrix[:3, 2] = normal
        transform_matrix[:3, 3] = point
        transform_matrix = transform_matrix @ self.center_transform
        return transform_matrix

def normalize(v):
    """Normalize a vector."""
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("Zero vector cannot be normalized.")
    return v / norm


def empty_refinement(transform, segmented_points, neighbors, label, source, target):
    score = len(neighbors[label])
    return transform, score


class Reverse_icp_for_segmentation:
    def __init__(self, icp_thresholds):
        self.icp_thresholds = icp_thresholds

    def refine(self, transform, segmented_points, neighbors, label, source, target):
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(segmented_points[ neighbors[label] ])
        
        transform = np.linalg.inv(transform)

        for threshold in self.icp_thresholds:
            reg_p2p = o3d.pipelines.registration.registration_icp(
                target, source, threshold, transform,
                o3d.pipelines.registration.TransformationEstimationPointToPoint())
            transform = reg_p2p.transformation

        transform = np.linalg.inv(reg_p2p.transformation)

        score = len(neighbors[label])

        return transform, score


class Rotational_icp_search_reversed_with_depth_check:
    def __init__(self, depth, depth_checker, background_distance, acceptance_distance, icp_thresholds, minimum_fitness):
        self.depth = depth
        self.depth_checker = depth_checker
        self.background_distance = background_distance
        self.acceptance_distance = acceptance_distance
        self.icp_thresholds = icp_thresholds
        self.minimum_fitness = minimum_fitness

    def refine(self, transform, segmented_points, neighbors, label, source, target):
        best_transform = np.eye(4)
        best_score = 0
        
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(segmented_points[ neighbors[label] ])

        for j in [0, 180]:
            for i in range(0, 351, 10):

                z_rot_transform = np.eye(4)
                z_rot_transform[:3, :3] = Rotation.from_rotvec([0, 0, np.pi*i/180.0]).as_matrix()
                
                y_rot_transform = np.eye(4)
                y_rot_transform[:3, :3] = Rotation.from_rotvec([np.pi*j/180.0, 0, 0]).as_matrix()

                transform_test = transform @ y_rot_transform @ z_rot_transform

                transform_test = np.linalg.inv(transform_test)
                for threshold in self.icp_thresholds:
                    reg_p2p = o3d.pipelines.registration.registration_icp(
                        target, source, threshold, transform_test,
                        o3d.pipelines.registration.TransformationEstimationPointToPoint(),)
                    transform_test = reg_p2p.transformation
                
                transform_test = np.linalg.inv(transform_test)

                if outside_azimuth(transform_test):
                    continue

                if reg_p2p.fitness < self.minimum_fitness:
                    continue

                depth_score = self.depth_checker.compute(transform_test, self.depth, self.background_distance, self.acceptance_distance)
                if depth_score > best_score:
                    best_score = depth_score
                    best_transform = transform_test

        return best_transform, best_score*len(neighbors[label])


def outside_azimuth(transform):
    z_axis = transform[:3, 2]
    z_angle = 180*np.arccos(-z_axis[2])/np.pi
    if(z_angle < -58 and z_axis[1] < 0):
        return True
    if(z_angle > 88 and z_axis[1] >= 0):
        return True
    return False


class C2f_icp:
    def __init__(self, icp_thresholds):
        self.icp_thresholds = icp_thresholds

    def refine(self, transform, segmented_points, neighbors, label, source, target):

        for threshold in self.icp_thresholds:
            reg_p2p = o3d.pipelines.registration.registration_icp(
                source, target, threshold, transform,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),)
            transform = reg_p2p.transformation

        score = len(neighbors[label])
        return transform, score
