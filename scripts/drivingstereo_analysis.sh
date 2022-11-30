#!/usr/bin/env bash
set -e
DATAPATH="/media/Data/datasets/DrivingStereo/"

python post_process/generate_statistic.py --logdir /mnt/Data2/liyan/GwcNet/checkpoints/drivingstereo/gwcnet-gc-elu-dropout-l1-logs-kl-5std-logspace-11bins-vkpretrained-lr1e-4 \
    --epochs 19 \
    --maxdisp 192 \
    --inliers 5 \
    --mask 'soft' \
    --bin_scale 'log' \
    --n_bins 11 \
    --uncert \
    --dataset 'drivingstereo'

python post_process/generate_statistic.py --logdir /mnt/Data2/liyan/GwcNet/checkpoints/drivingstereo/gwcnet-gc-elu-dropout-l1-logs-vkpretrained-lr1e-4 \
    --epochs 19 \
    --inliers 0 \
    --maxdisp 192 \
    --mask 'soft' \
    --bin_scale 'log' \
    --n_bins 11 \
    --uncert \
    --dataset 'drivingstereo'

python post_process/generate_statistic.py --logdir /mnt/Data2/liyan/GwcNet/checkpoints/drivingstereo/gwcnet-gc-sml1-lr1e-4/weather/ \
    --epochs 0 \
    --maxdisp 192 \
    --inliers 0 \
    --mask 'soft' \
    --bin_scale 'log' \
    --n_bins 11 \
    --dataset 'drivingstereo'

python post_process/generate_statistic.py --logdir /mnt/Data2/liyan/GwcNet/checkpoints/drivingstereo/gwcnet-gcs-kg-vkpt-lr1e-4/weather/ \
    --epochs 0 \
    --maxdisp 192 \
    --inliers 0 \
    --mask 'hard' \
    --bin_scale 'log' \
    --n_bins 11 \
    --uncert \
    --dataset 'drivingstereo'

python post_process/generate_statistic.py --logdir /mnt/Data2/liyan/GwcNet/checkpoints/drivingstereo/gwcnet-gcs-kg-5-vkpt-lr1e-4/weather/ \
    --epochs 0 \
    --maxdisp 192 \
    --inliers 5 \
    --mask 'hard' \
    --bin_scale 'log' \
    --n_bins 11 \
    --uncert \
    --dataset 'drivingstereo'

python post_process/generate_statistic.py --logdir /mnt/Data2/liyan/GwcNet/checkpoints/drivingstereo/gwcnet-gcs-uc-1std-log-11b-vkpt-lr1e-4/weather/ \
    --epochs 0 \
    --maxdisp 192 \
    --inliers 1 \
    --mask 'soft' \
    --bin_scale 'log' \
    --n_bins 11 \
    --uncert \
    --dataset 'drivingstereo'

python post_process/generate_statistic.py --logdir /mnt/Data2/liyan/GwcNet/checkpoints/drivingstereo/gwcnet-gcs-uc-3std-log-11b-vkpt-lr1e-4/weather/ \
    --epochs 0 \
    --maxdisp 192 \
    --inliers 3 \
    --mask 'soft' \
    --bin_scale 'log' \
    --n_bins 11 \
    --uncert \
    --dataset 'drivingstereo'

python post_process/generate_statistic.py --logdir /mnt/Data2/liyan/GwcNet/checkpoints/drivingstereo/gwcnet-gcs-uc-5std-log-11b-vkpt-lr1e-4/weather/ \
    --epochs 0 \
    --maxdisp 192 \
    --inliers 5 \
    --mask 'soft' \
    --bin_scale 'log' \
    --n_bins 11 \
    --uncert \
    --dataset 'drivingstereo'
