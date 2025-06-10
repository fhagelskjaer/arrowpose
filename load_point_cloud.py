import time
import numpy as np
import open3d as o3d
import cv2
from sklearn.neighbors import KDTree

def load_point_cloud(intrinsic, depth_scale, num_points, name, neighbors=20, rgb_name="", depth_trunc=1000.0, verbose=False):
    
    start_time_seconds = time.time()
 
    rgb_available = rgb_name != ""

    scene_pointcloud, depth = load_depth_image(intrinsic, depth_scale, name, rgb_name, depth_trunc, verbose)
    pointcloud, neighbor_idx_val, scene_pointcloud = preprocess_pointcloud(scene_pointcloud, num_points, neighbors, rgb_available, start_time_seconds, verbose)

    return pointcloud, neighbor_idx_val, scene_pointcloud, depth

def load_depth_image(intrinsic, depth_scale, name, rgb_name="", depth_trunc=1000.0, start_time_seconds=0, verbose=False):

    depth_image = cv2.imread(name, cv2.IMREAD_ANYDEPTH)
    depth = np.array(depth_image, np.float64)
    depth = depth * depth_scale
    depth = depth * 10

    depth = np.array(depth, np.uint16)
    depth_raw = o3d.geometry.Image(depth)

    if rgb_name != "":
        color_raw = o3d.io.read_image(rgb_name)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, depth_scale=10, depth_trunc=depth_trunc, convert_rgb_to_intensity=False)
        scene_pointcloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    else:
        scene_pointcloud = o3d.geometry.PointCloud.create_from_depth_image(depth_raw, intrinsic, depth_scale=10, depth_trunc=depth_trunc, project_valid_depth_only=True)
    
    if verbose:
        print('loading pc took %0.3f s' % (time.time() - start_time_seconds))

    depth = depth/10.0

    return scene_pointcloud, depth

def preprocess_pointcloud(scene_pointcloud, num_points, neighbors, rgb_available=False, start_time_seconds=0, verbose=False):

    pointcloud = np.asarray(scene_pointcloud.points)

    index_list = np.arange(len(pointcloud))
    # insert code to double point cloud if it is too small        
    while len(index_list) < num_points:
        index_list = np.array(list(index_list) + list(index_list))
    np.random.shuffle(index_list)
    pointcloud = pointcloud[ index_list, : ]

    pointcloud = pointcloud[ :num_points, : ]

    scene_pointcloud.points = o3d.utility.Vector3dVector(pointcloud) 
    if rgb_available:
        scene_pointcloud.colors = o3d.utility.Vector3dVector( np.asarray(scene_pointcloud.colors)[index_list, : ][ :num_points, : ] ) 
    
    scene_pointcloud.estimate_normals( o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=10), fast_normal_computation=True)
    scene_pointcloud.orient_normals_to_align_with_direction(orientation_reference=np.array([0.0, 0.0, -1.0]))

    pointcloud = np.concatenate([np.asarray(scene_pointcloud.points), np.asarray(scene_pointcloud.normals)], axis = 1)

    pointcloud = pointcloud.astype('float32')
    
    if verbose:
        print('pc processing took %0.3f s' % (time.time() - start_time_seconds))

    neighbor_idx = []
    for neighbor_size, point_layer_size, k in [(num_points,num_points//4, neighbors+1),(num_points//4,num_points//8, neighbors+1),(num_points//8,num_points//16,neighbors+1),(num_points//16,num_points//32,neighbors+1),(num_points//32,num_points//32,neighbors+1) ]:
        kdtree = KDTree( pointcloud[:neighbor_size,:3])
        neighbors = kdtree.query( pointcloud[:point_layer_size,:3], k=k, return_distance=False, dualtree=False, sort_results=False)
        neighbors = neighbors[:,1:] # to avoid itself
        neighbor_idx.append(neighbors)
    
    if verbose:
        print('idx computation took %0.3f s' % (time.time() - start_time_seconds))

    neighbor_idx_val = np.concatenate(neighbor_idx, axis=0)

    if verbose:
        print('concat took %0.3f s' % (time.time() - start_time_seconds))

    return np.array([pointcloud]), np.array([neighbor_idx_val]), scene_pointcloud
