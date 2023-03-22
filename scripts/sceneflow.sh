#!/usr/bin/env bash
set -e
DATAPATH="/media/datasets/SceneFlow/"
SAVEPATH="./checkpoints/sceneflow/"
#python main.py --dataset sceneflow \
#    --datapath $DATAPATH --trainlist ./filenames/sceneflow_train.txt --testlist ./filenames/sceneflow_test.txt \
#    --maxdisp 192 \
#    --zoom 0.5 \
#    --crop_w 448 \
#    --crop_h 284 \
#    --epochs 17 --lrepochs "10,12,14,16:2" \
#    --batch_size 8 \
#    --loss_type 'smooth_l1' \
#    --training \
#    --save_test \
#    --mask 'soft' \
#    --bin_scale 'log' \
#    --n_bins 11 \
#    --inliers 3 \
#    --model gwcnet-gc \
#    --logdir $SAVEPATH/gwc-gc-sml1-log-3std/ \
#    --test_batch_size 2 \
#    --device_id 0

python post_process/generate_statistic.py --logdir $SAVEPATH/gwc-gc-sml1-log-3std/ \
    --epochs 16 \
    --maxdisp 192 \
    --zoom 0.5 \
    --inliers 3 \
    --mask 'soft' \
    --bin_scale 'log' \
    --n_bins 11 \
    --dataset 'sceneflow'

#python main.py --dataset sceneflow \
#    --datapath $DATAPATH --trainlist ./filenames/sceneflow_train.txt --testlist ./filenames/sceneflow_test.txt \
#    --maxdisp 192 \
#    --zoom 0.5 \
#    --crop_w 448 \
#    --crop_h 284 \
#    --epochs 17 --lrepochs "10,12,14,16:2" \
#    --batch_size 8 \
#    --loss_type 'smooth_l1' \
#    --training \
#    --save_test \
#    --mask 'hard' \
#    --bin_scale 'log' \
#    --inliers 5 \
#    --model gwcnet-gc \
#    --logdir $SAVEPATH/gwc-gc-sml1-epe5/ \
#    --test_batch_size 2 \
#    --device_id 0

python post_process/generate_statistic.py --logdir $SAVEPATH/gwc-gc-sml1-epe5/ \
    --epochs 16 \
    --maxdisp 192 \
    --zoom 0.5 \
    --inliers 5 \
    --mask 'hard' \
    --dataset 'sceneflow'


#python main.py --dataset sceneflow \
#    --datapath $DATAPATH --trainlist ./filenames/sceneflow_train.txt --testlist ./filenames/sceneflow_test.txt \
#    --maxdisp 192 \
#    --zoom 0.5 \
#    --crop_w 448 \
#    --crop_h 284 \
#    --epochs 17 --lrepochs "10,12,14,16:2" \
#    --batch_size 8 \
#    --loss_type 'KG' \
#    --training \
#    --save_test \
#    --mask 'soft' \
#    --bin_scale 'log' \
#    --inliers 3 \
#    --model gwcnet-gcs \
#    --logdir $SAVEPATH/gwc-gcs-kg-log-3std/ \
#    --test_batch_size 2 \
#    --device_id 0
python post_process/generate_statistic.py --logdir $SAVEPATH/gwc-gcs-kg-log-3std/ \
    --epochs 16 \
    --maxdisp 192 \
    --zoom 0.5 \
    --inliers 3 \
    --mask 'soft' \
    --bin_scale 'log' \
    --n_bins 11 \
    --uncert \
    --dataset 'sceneflow'


#python main.py --dataset sceneflow \
#    --datapath $DATAPATH --trainlist ./filenames/sceneflow_train.txt --testlist ./filenames/sceneflow_test.txt \
#    --maxdisp 192 \
#    --zoom 0.5 \
#    --crop_w 448 \
#    --crop_h 284 \
#    --epochs 17 --lrepochs "10,12,14,16:2" \
#    --batch_size 8 \
#    --loss_type 'UC' \
#    --training \
#    --save_test \
#    --mask 'hard' \
#    --bin_scale 'log' \
#    --inliers 5 \
#    --model gwcnet-gcs \
#    --logdir $SAVEPATH/gwc-gcs-uc-epe5/ \
#    --test_batch_size 2 \
#    --device_id 0

python post_process/generate_statistic.py --logdir $SAVEPATH/gwc-gcs-uc-epe5/ \
    --epochs 16 \
    --maxdisp 192 \
    --zoom 0.5 \
    --inliers 5 \
    --mask 'hard' \
    --uncert \
    --dataset 'sceneflow'