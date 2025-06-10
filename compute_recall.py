import argparse
import json
import numpy as np

import pandas as pd

import math

def gtreadvis(gty, idx, max_idx):
    gt_poses = {}
    for key in gty.keys():
        pose_estimations_in_scene = []

        for i in range( len( gty[key] )):
            
            if len(pose_estimations_in_scene) >= max_idx:
                continue
        
            transform = np.eye(4)
            transform[:3,3] = np.array(gty[key][i]['cam_t_m2c'])
            transform[:3,:3] = np.array( gty[key][i]['cam_R_m2c'] ).reshape( (3,3) )
            if( int(gty[key][i]['obj_id'])  == int(idx)):
                
                pose_estimations_in_scene.append( transform )
                
        gt_poses[key] = pose_estimations_in_scene
    return gt_poses


def dist_and_angle(pose, acc_pose, angle_test=15):
    dot_product = np.dot(pose[:3, 2], acc_pose[:3, 2]) 
    if dot_product >= 1.0:
        angle = 0
    elif dot_product <= -1.0:
        angle = -180.0
    else:
        angle = (180/np.pi)*np.arccos(dot_product)
    
    if angle < angle_test:
        return math.dist(pose[:3, 3], acc_pose[:3, 3])
    else:
        return 1000


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')
    parser.add_argument('--gt_file', type=str, default='',
                        help='reference to the rgb image')
    parser.add_argument('--result_file', type=str, default='',
                        help='reference to the rgb image')
    parser.add_argument('--obj_idx', type=int, default=1, metavar='N',
                        help='Number of output channels in the trained network')
    args = parser.parse_args()


    distance_threshold = 50

    gt_file = args.gt_file
    result_file = args.result_file

    scene_idx = str(int(gt_file.split('/')[-2]))

    max_idx = 40

    if scene_idx == "1":
        max_idx = 15
    if scene_idx == "3":
        max_idx = 10

    gt_file = json.load(open(gt_file))
    gt_poses = gtreadvis(gt_file, args.obj_idx, max_idx=max_idx)

    pose_estimations = np.loadtxt(result_file, delimiter=",", dtype=str)
    
    # first limit by object
    pose_estimations = pose_estimations[ pose_estimations[:,2] == str(args.obj_idx), : ]
    
    # then limit by scene
    pose_estimations = pose_estimations[ pose_estimations[:,0] == scene_idx, : ]
    # then create a set of all images
    all_scene_ids = list(set(pose_estimations[:,1]))
    
    all_scene_ids = sorted(all_scene_ids, key=lambda scene: int(scene))
    
    pose_estimations_sorted = sorted(pose_estimations, key=lambda pose_dict: -float(pose_dict[3]))

    if args.obj_idx == 2:
        max_pred = len(gt_poses['0'])
    if args.obj_idx == 1:
        max_pred = 10
    
    sum_of_possible_gt = 0
    for scene_id in all_scene_ids:
        sum_of_possible_gt += max_pred

    recall_rate = 0

    for scene_id in all_scene_ids:

        for gt_pose in gt_poses[scene_id]:
            
            dist_list = []

            for pose_estimate in pose_estimations_sorted:
            
                if len(dist_list) == max_pred:
                    break

                if not scene_id == pose_estimate[1]:
                    continue
                
                transform = np.eye(4)
                transform[:3,:3] = np.reshape( np.array(pose_estimate[4].split(' '), float), (3,3))
                transform[:3,3] = np.array(pose_estimate[5].split(' '), float )

                dist_list.append( dist_and_angle(np.array(gt_pose), transform) )

            dist_list.sort()
            if dist_list[0] < distance_threshold:
                recall_rate += 1

    print(recall_rate, sum_of_possible_gt, recall_rate/sum_of_possible_gt )

if __name__ == "__main__":
    main()

