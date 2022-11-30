#!/usr/bin/env bash
set -e
DATAPATH="/media/Data/datasets/DrivingStereo/"


python main.py --dataset drivingstereo \
    --datapath $DATAPATH --trainlist ./filenames/drivingstereo_train.txt --testlist ./filenames/drivingstereo_test_weather.txt \
    --maxdisp 192 \
    --epochs 1 --lrepochs " 10,13,18:2" \
    --batch_size 8 \
    --lr 0.0001 \
    --loss_type 'smooth_l1' \
    --save_test \
    --bin_scale 'log' \
    --n_bins 11 \
    --inliers 5 \
    --model gwcnet-gc \
    --loadckpt ./checkpoints/vkitti2/gwcnet-gc-sml1-lr1e-4/checkpoint_000020.ckpt \
    --logdir ./checkpoints/drivingstereo/gwcnet-gc-sml1-lr1e-4/weather/ \
    --test_batch_size 1

python main.py --dataset drivingstereo \
    --datapath $DATAPATH --trainlist ./filenames/drivingstereo_train.txt --testlist ./filenames/drivingstereo_test_weather.txt \
    --maxdisp 192 \
    --epochs 1 --lrepochs " 10,13,18:2" \
    --batch_size 8 \
    --lr 0.0001 \
    --loss_type 'KG' \
    --save_test \
    --mask 'hard' \
    --bin_scale 'log' \
    --n_bins 11 \
    --inliers 5 \
    --model gwcnet-gcs \
    --loadckpt ./checkpoints/vkitti2/gwcnet-gc-elu-dropout-l1-logs-inliers-1re-4/checkpoint_000018.ckpt \
    --logdir ./checkpoints/drivingstereo/gwcnet-gcs-kg-5-vkpt-lr1e-4/weather/ \
    --test_batch_size 1

python main.py --dataset drivingstereo \
    --datapath $DATAPATH --trainlist ./filenames/drivingstereo_train.txt --testlist ./filenames/drivingstereo_test_weather.txt \
    --maxdisp 192 \
    --epochs 1 --lrepochs " 10,13,18:2" \
    --batch_size 8 \
    --lr 0.0001 \
    --loss_type 'KG' \
    --save_test \
    --bin_scale 'log' \
    --n_bins 11 \
    --inliers 0 \
    --model gwcnet-gcs \
    --loadckpt ./checkpoints/vkitti2/gwcnet-gc-elu-dropout-l1-logs-lr1e-4/checkpoint_000029.ckpt \
    --logdir ./checkpoints/drivingstereo/gwcnet-gcs-kg-vkpt-lr1e-4/weather/ \
    --test_batch_size 1

python main.py --dataset drivingstereo \
    --datapath $DATAPATH --trainlist ./filenames/drivingstereo_train.txt --testlist ./filenames/drivingstereo_test_weather.txt \
    --maxdisp 192 \
    --epochs 1 --lrepochs " 10,13,18:2" \
    --batch_size 8 \
    --lr 0.0001 \
    --loss_type 'UC' \
    --save_test \
    --bin_scale 'log' \
    --n_bins 11 \
    --inliers 1 \
    --model gwcnet-gcs \
    --loadckpt ./checkpoints/vkitti2/gwcnet-gc-elu-dropout-l1-logs-kl-1std-lr1e-4/checkpoint_000059.ckpt \
    --logdir ./checkpoints/drivingstereo/gwcnet-gcs-uc-1std-log-11b-vkpt-lr1e-4/weather/ \
    --test_batch_size 1

python main.py --dataset drivingstereo \
    --datapath $DATAPATH --trainlist ./filenames/drivingstereo_train.txt --testlist ./filenames/drivingstereo_test_weather.txt \
    --maxdisp 192 \
    --epochs 1 --lrepochs " 10,13,18:2" \
    --batch_size 8 \
    --lr 0.0001 \
    --loss_type 'UC' \
    --save_test \
    --bin_scale 'log' \
    --n_bins 11 \
    --inliers 5 \
    --model gwcnet-gcs \
    --loadckpt ./checkpoints/vkitti2/gwcnet-gc-elu-dropout-l1-logs-kl-5std-logspace-lr1e-4/checkpoint_000031.ckpt \
    --logdir ./checkpoints/drivingstereo/gwcnet-gcs-uc-5std-log-11b-vkpt-lr1e-4/weather/ \
    --test_batch_size 1