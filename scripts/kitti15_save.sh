#!/usr/bin/env bash
set -x
DATAPATH="/mnt/Data2/datasets/KITTI-2015/data_scene_flow/"
SAVEPATH="/mnt/Data2/liyan/GwcNet/"
python save_disp.py \
  --datapath $DATAPATH \
  --testlist ./filenames/kitti15_test.txt \
  --model gwcnet-gcs --loadckpt $SAVEPATH/checkpoints/kitti15/gwcnet-gcs-uc-3std-11b/checkpoint_000396.ckpt
