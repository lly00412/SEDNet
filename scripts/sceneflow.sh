#!/usr/bin/env bash
set -e
DATAPATH="/media/Data/datasets/SceneFlow/"
SAVEPATH="/mnt/Data2/liyan/GwcNet/"
python main.py --dataset sceneflow \
    --datapath $DATAPATH --trainlist ./filenames/sceneflow_train.txt --testlist ./filenames/sceneflow_test.txt \
    --maxdisp 192 \
    --epochs 30 --lrepochs "7,12,14,20:2" \
    --batch_size 32 \
    --lr 0.001 \
    --resume \
    --training \
    --model gwcnet-gcs \
    --loss_type 'UC' \
    --save_test \
    --bin_scale 'log' \
    --n_bins 11 \
    --inliers 3 \
    --logdir $SAVEPATH/checkpoints/sceneflow/gwcnet-gcs-uc-3std-11b \
    --batch_size 2 \
    --test_batch_size 2 \
    --device_id 0 1