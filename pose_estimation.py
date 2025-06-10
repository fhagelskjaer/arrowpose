import glob
import time
import copy
import argparse

import numpy as np
import torch
import torch.nn as nn
import open3d as o3d

import arrowpose
from model import LottoNetDet
from depthcheck import DepthCheck
import vis
import load_point_cloud


def print_points(poses):
    print("points_possibly_inside_points = np.array([", end =" ")
    for center in poses:
        print('[' +str(center[0])+","+str(center[1])+","+str(center[2])+'],', end =" ")
    print("])") 
    return

def test(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    
    np.random.seed(0)
    np.set_printoptions(suppress=True)

    intrinsic = o3d.camera.PinholeCameraIntrinsic()

    w,h,fx,fy,cx,cy,depth_scale = args.intrinsics.split(",")
    w,h,fx,fy,cx,cy,depth_scale = int(w),int(h),float(fx),float(fy),float(cx),float(cy),float(depth_scale)

    intrinsic.set_intrinsics(w,h,fx,fy,cx,cy)

    channels = args.channels
    search_seg_idx = args.search_seg_idx

    cad_model_string = args.cad
    source_cad = o3d.io.read_triangle_mesh(cad_model_string)
    source = source_cad.sample_points_poisson_disk(2048) 

    output_size = args.num_points//32

    model = LottoNetDet(args.k, 0, output_channels=channels).to(device)

    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_root, map_location=device))
    model = model.eval()
   
    depth_checker = DepthCheck(cad_model_string,w,h,fx,fy,cx,cy)

    data = torch.randn((1,6,args.num_points), dtype=torch.float32)
    idx = torch.zeros((1,args.num_points//2,args.k), dtype=torch.int64)

    model(data, idx, device)

    name_list = glob.glob(args.depth_name)
    name_list.sort()
    if len(name_list) > 1:
        rgb_name = ""
    else:
        rgb_name = args.rgb_name

    arrowPoseEstimator = arrowpose.ArrowPose(cluster_radius=args.cluster_radius,
                                           min_num_point_in_cluster=args.min_num_point_center,
                                           source=source,
                                           compute_transform_matrix=arrowpose.TransformMatrix().compute_transform_matrix_z,
                                           )

    for name in name_list:

        start_time_seconds = time.time()

        pointcloud, neighbor_idx_val, pcd_o3d, depth = load_point_cloud.load_point_cloud(intrinsic,depth_scale,args.num_points,name,args.k,rgb_name,depth_trunc=1600.0,verbose=True)

        resized_pointcloud = np.array(pointcloud[0,:output_size,:3])

        if args.fig:
            vis.visualize_pyramid(pcd_o3d)
            vis.visualize_neighborhood(test_point_idx_list=[1105,1204,105,1206], pointcloud_o3d=pcd_o3d, idx_val=neighbor_idx_val)

        print('Data collection took %0.3f s' % (time.time() - start_time_seconds))

        data, idx = torch.from_numpy(pointcloud), torch.from_numpy(neighbor_idx_val)
        data, idx = data.to(device), idx.to(device)
            
        data = data.permute(0, 2, 1)

        seg_pred, cen_pred, top_pred = model(data, idx, device)
            
        print('Model processing took %0.3f s' % (time.time() - start_time_seconds))
            

        if args.icp_type == arrowpose.ICPTYPE.ONLY_POINTS.value:
            refinement = None
        elif args.icp_type == arrowpose.ICPTYPE.NONE.value:
            refinement = arrowpose.empty_refinement
        elif args.icp_type == arrowpose.ICPTYPE.ICP.value:
            refinement = arrowpose.Reverse_icp_for_segmentation([args.icp_threshold, 5]).refine
        elif args.icp_type == arrowpose.ICPTYPE.ICPROT.value:
            refinement = arrowpose.Rotational_icp_search_reversed_with_depth_check(depth, depth_checker, 0.043, 0.012, [args.icp_threshold, 5], 0.8).refine # (transform, segmented_points, neighbors, label, self.source, self.icp_threshold, )
        elif args.icp_type == arrowpose.ICPTYPE.CLASSIC.value:
            refinement = arrowpose.C2f_icp([args.icp_threshold, 5]).refine


        accepted_poses = arrowPoseEstimator.compute_poses(search_seg_idx=search_seg_idx,
                                                            seg_pred=seg_pred,
                                                            cen_pred=cen_pred,
                                                            top_pred=top_pred,
                                                            pointcloud_o3d=pcd_o3d,
                                                            resized_pointcloud=resized_pointcloud, 
                                                            refinement=refinement, 
                                                            visu=args.visu, 
                                                            fig=args.fig)
        
        print('Computing ICP took %0.3f s' % (time.time() - start_time_seconds))
        
        if args.icp_type == arrowpose.ICPTYPE.ONLY_POINTS.value:
            print_points(accepted_poses)
            return

        if args.bop_out != "":
            for transform_info in accepted_poses:
                transform, score = transform_info['t'], transform_info['s']
                with open(args.bop_out, 'a') as f:
                    f.write(  str(int(name.split('/')[-3])) + "," + str(int(name.split('/')[-1].split('.')[0])) + "," +  str(search_seg_idx) + "," + str(score) + "," + ' '.join( [ str(c) for c in transform[:3,:3].flatten().tolist() ] ) + "," + ' '.join( [ str(c) for c in transform[:3,3].flatten().tolist() ] )  + "," + "-1\n" )


        if args.visu:
            max_score = np.max([ transform_info['s'] for transform_info in accepted_poses])
                
            to_draw = []
            for transform_info in accepted_poses:

                source_temp = copy.copy(source_cad)
                transform, score = transform_info['t'], transform_info['s']
                
                source_temp.paint_uniform_color([1-(1.0*score/max_score), 1.0*score/max_score, 0])
                source_temp.transform(transform)
                to_draw.append(source_temp)

            o3d.visualization.draw_geometries([pcd_o3d] + to_draw, point_show_normal=False)    

            if args.fig:
                o3d.visualization.draw_geometries([pcd_o3d], point_show_normal=False)    
                o3d.visualization.draw_geometries(to_draw, point_show_normal=False)    


def main():
    parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')

    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--num_points', type=int, default=32768,
                        help='num of points to use')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_root', type=str, default='', metavar='N',
                        help='Pretrained model root')
    parser.add_argument('--visu', type=bool, default=False,
                        help='visualize the model')
    parser.add_argument('--fig', type=bool, default=False,
                        help='visualize the model for paper figures')
                        
    parser.add_argument('--intrinsics', type=str, default='640,480,550.0,540.0,316.0,244.0,1', metavar='N',
                        help='camera intrinsic, w,h,fx,fy,cx,cy,depth_scale')
    parser.add_argument('--channels', type=int, default=2, metavar='N',
                        help='Number of output channels in the trained network')
    parser.add_argument('--search_seg_idx', type=int, default=1, metavar='N',
                        help='Number of output channels in the trained network')
    parser.add_argument('--depth_name', type=str, default='',
                        help='reference to the depth image')
    parser.add_argument('--rgb_name', type=str, default='',
                        help='reference to the rgb image')                        
    parser.add_argument('--cad', type=str, default='',
                        help='reference to the rgb image')  

    parser.add_argument('--cluster_radius', type=int, default=10, metavar='N',
                    help='distance threshold cluster radius')
    parser.add_argument('--min_num_point_center', type=int, default=3, metavar='N',
                    help='minimum number of neighbors for accepting detection')
    parser.add_argument('--icp_threshold', type=int, default=10, metavar='N',
                    help='distance threshold for ICP')
    
    parser.add_argument('--icp_type', type=int, default=1,
                        help='what type of icp to perform, 0 only returns center points')

    parser.add_argument('--bop_out', type=str, default='',
                        help='output csv file for the bop format')  

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        print(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        print('Using CPU')

    test(args)


if __name__ == "__main__":
    main()
