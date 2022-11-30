#!/usr/bin/env bash
set -e
DATAPATH="../datasets/SceneFlow/"
#python post_process/generate_statistic.py --logdir ./LAF/checkpoints/scenflow/gwcnet-gc \
#    --epochs 16 \
#    --inliers 0 \
#    --bin_scale 'log' \
#    --n_bins 11

#python post_process/generate_statistic.py --logdir ./checkpoints/sceneflow/gwcnet-gc-elu-dropout-l1-logs-lr1e-4 \
#    --epochs 22 \
#    --inliers 0 \
#    --bin_scale 'log' \
#    --n_bins 11

#python post_process/generate_statistic.py --logdir ./checkpoints/sceneflow/gwcnet-gc-elu-dropout-l1-logs-inliers-lr1e-4 \
#    --epochs 29 \
#    --inliers 0 \
#    --bin_scale 'log' \
#    --n_bins 11

#python post_process/generate_statistic.py --logdir ./checkpoints/sceneflow/gwcnet-gc-elu-dropout-l1-logs-kl-3std-linespace-lr1e-3 \
#    --epochs 15 \
#    --inliers 3 \
#    --bin_scale 'line' \
#    --n_bins 11
#
#python post_process/generate_statistic.py --logdir ./checkpoints/sceneflow/gwcnet-gc-elu-dropout-l1-logs-kl-3std-logspace-lr1e-3 \
#    --epochs 15 \
#    --inliers 3 \
#    --bin_scale 'log' \
#    --n_bins 11

#python post_process/generate_statistic.py --logdir ./checkpoints/sceneflow/gwcnet-gc-elu-dropout-l1-logs-kl-3std-logspace-20bins-lr1e-3 \
#    --epochs 15 \
#    --inliers 3 \
#    --bin_scale 'log' \
#    --n_bins 20
#
#python post_process/generate_statistic.py --logdir ./checkpoints/sceneflow/gwcnet-gc-elu-dropout-l1-logs-kl-3std-logspace-50bins-lr1e-3 \
#    --epochs 15 \
#    --inliers 3 \
#    --bin_scale 'log' \
#    --n_bins 50
#
#python post_process/generate_statistic.py --logdir ./checkpoints/sceneflow/gwcnet-gc-elu-dropout-l1-logs-kl-3std-logspace-11bins-all-lr1e-3 \
#    --epochs 15 \
#    --inliers 0 \
#    --bin_scale 'log' \
#    --n_bins 11

python post_process/generate_statistic.py --logdir ./checkpoints/sceneflow/gwcnet-gc \
    --epochs 16 \
    --inliers 0 \
    --bin_scale 'log' \
    --n_bins 11 \
    --dataset 'sceneflow'


python post_process/generate_statistic.py --logdir ./checkpoints/sceneflow/gwcnet-gc-elu-dropout-l1-logs-lr1e-4 \
    --epochs 22 \
    --inliers 0 \
    --mask 'hard' \
    --bin_scale 'log' \
    --n_bins 11 \
    --dataset 'sceneflow' \
    --uncert

python post_process/generate_statistic.py --logdir ./checkpoints/sceneflow/gwcnet-gc-elu-dropout-l1-logs-inliers-lr1e-4 \
    --epochs 29 \
    --inliers 5 \
    --mask 'hard' \
    --bin_scale 'log' \
    --n_bins 11 \
    --dataset 'sceneflow' \
    --uncert

python post_process/generate_statistic.py --logdir ./checkpoints/sceneflow/gwcnet-gc-elu-dropout-l1-logs-kl-3std-linespace-lr1e-3 \
    --epochs 15 \
    --inliers 3 \
    --mask 'soft' \
    --bin_scale 'line' \
    --n_bins 11 \
    --dataset 'sceneflow' \
    --uncert

python post_process/generate_statistic.py --logdir ./checkpoints/sceneflow/gwcnet-gc-elu-dropout-l1-logs-kl-3std-logspace-lr1e-3 \
    --epochs 15 \
    --inliers 3 \
    --mask 'soft' \
    --bin_scale 'log' \
    --n_bins 11 \
    --dataset 'sceneflow' \
    --uncert

python post_process/generate_statistic.py --logdir ./checkpoints/sceneflow/gwcnet-gc-elu-dropout-l1-logs-kl-3std-logspace-20bins-lr1e-3 \
    --epochs 15 \
    --inliers 3 \
    --mask 'soft' \
    --bin_scale 'log' \
    --n_bins 20 \
    --dataset 'sceneflow' \
    --uncert

python post_process/generate_statistic.py --logdir ./checkpoints/sceneflow/gwcnet-gc-elu-dropout-l1-logs-kl-3std-logspace-50bins-lr1e-3 \
    --epochs 15 \
    --inliers 3 \
    --mask 'soft' \
    --bin_scale 'log' \
    --n_bins 50 \
    --dataset 'sceneflow' \
    --uncert

python post_process/generate_statistic.py --logdir ./LAF/checkpoints/scenflow/gwcnet-gc/ \
    --epochs 16 \
    --inliers 0 \
    --bin_scale 'log' \
    --n_bins 11 \
    --dataset 'sceneflow' \
    --conf
