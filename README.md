## ArrowPose

Visualize:

    python pose_estimation.py --model_root outputs/icbin/models/model.t7 --cluster_radius 25 --icp_threshold 17 --icp_type 2 --k 20 --num_points 65536 --search_seg_idx 1 --channels 3 --cad "../bop/icbin/models/obj_000001.ply" --depth_name "../bop/icbin/test/000001/depth/000020.png" --rgb_name "../bop/icbin/test/000001/rgb/000020.png" --visu True

    python pose_estimation.py --model_root outputs/icbin/models/model.t7 --cluster_radius 25 --icp_threshold 17 --icp_type 3 --k 20 --num_points 65536 --search_seg_idx 2 --channels 3 --cad "../bop/icbin/models/obj_000002.ply" --depth_name "../bop/icbin/test/000001/depth/000020.png" --rgb_name "../bop/icbin/test/000001/rgb/000020.png" --visu True

Run all:

    bash run_experiments.sh

    bash run_eval.sh

