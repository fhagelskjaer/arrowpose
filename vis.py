import copy
import open3d as o3d
import numpy as np

def visualize_pyramid(pointcloud_o3d):
    num_points = np.asarray(pointcloud_o3d.points).shape[0]
    obj_colors_rgb = np.array(pointcloud_o3d.colors)
    obj_colors =  np.array(pointcloud_o3d.colors)
    mean_color_value = np.mean(obj_colors_rgb, axis=1)
    obj_colors[:,0] = mean_color_value
    obj_colors[:,1] = mean_color_value
    obj_colors[:,2] = mean_color_value
    downscalings = [num_points//32]
    colors_float = [[0,1,0]]
    points = np.array(pointcloud_o3d.points)
    for scale_index, number_in_scale in enumerate(downscalings):
        obj_colors[:int(number_in_scale), :] = colors_float[scale_index]    
        points[:int(number_in_scale),2] -= 2
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(points)
    pcd_o3d.colors = o3d.utility.Vector3dVector(np.array(obj_colors))
    o3d.visualization.draw_geometries([pcd_o3d], point_show_normal=False)    

def visualize_neighborhood(test_point_idx_list, pointcloud_o3d, idx_val):
    num_point = np.asarray(pointcloud_o3d.points).shape[0]
    dim2, dim3, dim4, dim5 = num_point//4, num_point//8, num_point//16, num_point//32
    idx1 = idx_val[:, :dim2, :]
    idx2 = idx_val[:, dim2:(dim2+dim3), :]
    idx3 = idx_val[:, (dim2+dim3):(dim2+dim3+dim4), :]
    idx4 = idx_val[:, (dim2+dim3+dim4):(dim2+dim3+dim4+dim5), :]
    idx5 = idx_val[:, (dim2+dim3+dim4+dim5):(dim2+dim3+dim4+dim5+dim5), :]
    obj_colors_rgb = np.array(pointcloud_o3d.colors)
    obj_colors =  np.array(pointcloud_o3d.colors)
    mean_color_value = np.mean(obj_colors_rgb, axis=1)*0.7
    obj_colors[:,0] = mean_color_value
    obj_colors[:,1] = mean_color_value
    obj_colors[:,2] = mean_color_value
    idx_list = [idx5, idx4, idx3, idx2, idx1]
    neighbor_list = test_point_idx_list
    for idx_neighbors in idx_list:
        new_neighbor_list = []    
        for neighbor in neighbor_list:
            for i in range(20):
                new_neighbor_list.append( idx_neighbors[0,neighbor,i] )
        neighbor_list = new_neighbor_list
    for neighbor in neighbor_list:
        obj_colors[neighbor,2] = 1
    for neighbor in test_point_idx_list:
        obj_colors[neighbor,:] = [0,1,0]
        for i in range(20):
            obj_colors[idx1[0,neighbor,i],:] = [0,1,0]
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = pointcloud_o3d.points
    pcd_o3d.colors = o3d.utility.Vector3dVector(np.array(obj_colors))
    o3d.visualization.draw_geometries([pcd_o3d], point_show_normal=False)    

def visualize_prediction(pcd_o3d_show, segmented_points, cen_pred_np_seg, top_pred_np_seg, fig):
    pcd_o3d_cen = o3d.geometry.PointCloud()
    pcd_o3d_cen.points = o3d.utility.Vector3dVector(segmented_points-np.array(cen_pred_np_seg))
    pcd_o3d_top = o3d.geometry.PointCloud()
    pcd_o3d_top.points = o3d.utility.Vector3dVector(segmented_points-np.array(top_pred_np_seg))
    pcd_o3d_cen.paint_uniform_color([0, 1.0, 0])
    pcd_o3d_top.paint_uniform_color([1.0, 0, 0])
    o3d.visualization.draw_geometries([pcd_o3d_show, pcd_o3d_cen, pcd_o3d_top], point_show_normal=False)
    if fig:
        o3d.visualization.draw_geometries([pcd_o3d_show], point_show_normal=False)
        o3d.visualization.draw_geometries([pcd_o3d_cen], point_show_normal=False)
        o3d.visualization.draw_geometries([pcd_o3d_top], point_show_normal=False)

def visualize_segmentation(pcd_o3d_show, neighbors, unique_labels, center_points, segmented_points, colors_float, fig=False):
    obj_colors = np.zeros((len(neighbors), 3))
    for p_i in range(len(neighbors)):
        for label_index, label in enumerate(unique_labels):    
            if( p_i in neighbors[label] ):
                obj_colors[p_i] = colors_float[ label_index ]
                break
    pcd_o3d_cen = o3d.geometry.PointCloud()
    pcd_o3d_cen.points = o3d.utility.Vector3dVector(center_points)
    pcd_o3d_cen.colors = o3d.utility.Vector3dVector(np.array(obj_colors))
    o3d.visualization.draw_geometries([pcd_o3d_show, pcd_o3d_cen], point_show_normal=False)
    segmented_points_vis = np.array(segmented_points)
    segmented_points_vis[:,2] -= 3 
    pcd_o3d_seg = o3d.geometry.PointCloud()
    pcd_o3d_seg.points = o3d.utility.Vector3dVector(segmented_points_vis)
    pcd_o3d_seg.colors = o3d.utility.Vector3dVector(np.array(obj_colors))
    o3d.visualization.draw_geometries([pcd_o3d_show, pcd_o3d_seg], point_show_normal=False)
    if fig:
        o3d.visualization.draw_geometries([pcd_o3d_show], point_show_normal=False)
        o3d.visualization.draw_geometries([pcd_o3d_cen], point_show_normal=False)
        o3d.visualization.draw_geometries([pcd_o3d_seg], point_show_normal=False)

def visualize_pose_estimation(pointcloud_o3d, source, transform_list, colors_float, fig=False):
    to_draw = []
    for label_index, transform_info in enumerate(transform_list):
        source_temp = copy.copy(source)
        transform = transform_info['t']
        source_temp.paint_uniform_color(colors_float[label_index])
        source_temp.transform(transform)
        to_draw.append(source_temp)
    o3d.visualization.draw_geometries([pointcloud_o3d] + to_draw, point_show_normal=False)
    if fig:
        o3d.visualization.draw_geometries([pointcloud_o3d], point_show_normal=False)
        o3d.visualization.draw_geometries(to_draw, point_show_normal=False)

def visualize_arrow(pointcloud_o3d, transform_list, colors_float, fig=False):
    to_draw = []
    
    for label_index, transform_info in enumerate(transform_list):
        transform = transform_info['t']
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=5.0,
            cone_radius=10,
            cylinder_height=50,
            cone_height=15
        )
        arrow.transform(transform)
        arrow.paint_uniform_color(colors_float[label_index])
        to_draw.append(arrow)

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=12)
        sphere.transform(transform)
        sphere.paint_uniform_color(colors_float[label_index])
        to_draw.append(sphere)

    o3d.visualization.draw_geometries([pointcloud_o3d] + to_draw, point_show_normal=False)
    
    if fig:
        o3d.visualization.draw_geometries([pointcloud_o3d], point_show_normal=False)
        o3d.visualization.draw_geometries(to_draw, point_show_normal=False)
