#!/usr/bin/env bash
set -x
DATAPATH="/mnt/Data2/datasets/KITTI-2015/data_scene_flow/"
SAVEPATH="/mnt/Data2/liyan/GwcNet/"
python main.py --dataset kitti \
    --datapath $DATAPATH --trainlist ./filenames/kitti15_train.txt --testlist ./filenames/kitti15_val.txt \
    --epochs 500 --lrepochs "184,300:5" \
    --maxdisp 192 \
    --model gwcnet-gcs \
    --batch_size 8 \
    --resume \
    --training \
    --loss_type 'UC' \
    --save_test \
    --bin_scale 'log' \
    --n_bins 11 \
    --inliers 5 \
    --logdir $SAVEPATH/checkpoints/kitti15/gwcnet-gcs-uc-3std-11b/ \
    --batch_size 2 \
    --test_batch_size 1