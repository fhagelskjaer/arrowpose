#!/bin/bash

inputfile="results/icbin.csv"

python compute_recall.py --gt_file ../bop/icbin/test/000001/scene_gt.json --obj_idx 1 --result_file $inputfile
python compute_recall.py --gt_file ../bop/icbin/test/000003/scene_gt.json --obj_idx 1 --result_file $inputfile
python compute_recall.py --gt_file ../bop/icbin/test/000002/scene_gt.json --obj_idx 2 --result_file $inputfile
python compute_recall.py --gt_file ../bop/icbin/test/000003/scene_gt.json --obj_idx 2 --result_file $inputfile

