#!/bin/bash

outputfile="results/icbin.csv"

model="icbin"

python pose_estimation.py --model_root outputs/$model/models/model.t7 --cluster_radius 25 --icp_threshold 17 --icp_type 2 --k 20 --num_points 65536 --bop_out $outputfile --search_seg_idx 1 --channels 3 --cad "../bop/icbin/models/obj_000001.ply" --depth_name "../bop/icbin/test/000001/depth/*.png"

python pose_estimation.py --model_root outputs/$model/models/model.t7 --cluster_radius 25 --icp_threshold 17 --icp_type 2 --k 20 --num_points 65536 --bop_out $outputfile --search_seg_idx 1 --channels 3 --cad "../bop/icbin/models/obj_000001.ply" --depth_name "../bop/icbin/test/000003/depth/*.png"

python pose_estimation.py --model_root outputs/$model/models/model.t7 --min_num_point_center 6 --cluster_radius 25 --icp_threshold 17 --icp_type 3 --k 20 --num_points 65536 --bop_out $outputfile --search_seg_idx 2 --channels 3 --cad "../bop/icbin/models/obj_000002.ply" --depth_name "../bop/icbin/test/000002/depth/*.png"

python pose_estimation.py --model_root outputs/$model/models/model.t7 --min_num_point_center 6 --cluster_radius 25 --icp_threshold 17 --icp_type 3 --k 20 --num_points 65536 --bop_out $outputfile --search_seg_idx 2 --channels 3 --cad "../bop/icbin/models/obj_000002.ply" --depth_name "../bop/icbin/test/000003/depth/*.png"
