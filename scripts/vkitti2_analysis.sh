#!/usr/bin/env bash
set -e
DATAPATH="/media/Data/datasets/vkitti2/"

#python main.py --dataset vkitti2 \
#    --datapath $DATAPATH --trainlist ./filenames/vkitti2_train_wS6.txt --testlist ./filenames/vkitti2_test_wS6.txt \
#    --zoom 1.0 \
#    --maxdisp 192 \
#    --epochs 31 --lrepochs " 7,20,30,40,50:5" \
#    --batch_size 8 \
#    --lr 0.0001 \
#    --resume \
#    --loss_type 'KG' \
#    --save_test \
#    --model gwcnet-gcs \
#    --logdir ./checkpoints/vkitti2/gwcnet-gc-elu-dropout-l1-logs-lr1e-4 \
#    --test_batch_size 1


#python post_process/generate_statistic.py --logdir ./checkpoints/vkitti2/gwcnet-gc-elu-dropout-l1-logs-lr1e-4 \
#    --epochs 30 \
#    --inliers 0 \
#    --mask 'soft' \
#    --bin_scale 'log' \
#    --n_bins 11 \
#    --maxdisp 192 \
#    --uncert \
#    --dataset 'vkitti2'
#
#python post_process/generate_statistic.py --logdir ./checkpoints/vkitti2/gwcnet-gc-elu-dropout-l1-logs-kl-1std-lr1e-4 \
#    --epochs 59 \
#    --inliers 1 \
#    --mask 'soft' \
#    --bin_scale 'log' \
#    --n_bins 11 \
#    --maxdisp 192 \
#    --uncert \
#    --dataset 'vkitti2'
#
#python post_process/generate_statistic.py --logdir ./checkpoints/vkitti2/gwcnet-gc-elu-dropout-l1-logs-kl-3std-lr1e-4 \
#    --epochs 26 \
#    --inliers 3 \
#    --mask 'soft' \
#    --bin_scale 'log' \
#    --n_bins 11 \
#    --maxdisp 192 \
#    --uncert \
#    --dataset 'vkitti2'
#
#python post_process/generate_statistic.py --logdir ./checkpoints/vkitti2/gwcnet-gc-elu-dropout-l1-logs-kl-5std-logspace-lr1e-4 \
#    --epochs 32 \
#    --inliers 5 \
#    --mask 'soft' \
#    --bin_scale 'log' \
#    --n_bins 11 \
#    --maxdisp 192 \
#    --uncert \
#    --dataset 'vkitti2'


#python post_process/generate_statistic.py --logdir ./checkpoints/vkitti2/gwcnet-gc-sml1-lr1e-4 \
#    --epochs 21 \
#    --inliers 0 \
#    --maxdisp 192 \
#    --mask 'soft' \
#    --bin_scale 'log' \
#    --n_bins 11 \
#    --dataset 'vkitti2'
#
#python post_process/generate_statistic.py --logdir ./checkpoints/vkitti2/gwcnet-gc-elu-dropout-l1-logs-inliers-1re-4 \
#    --epochs 18 \
#    --inliers 5 \
#    --mask 'hard' \
#    --bin_scale 'log' \
#    --n_bins 11 \
#    --maxdisp 192 \
#    --uncert \
#    --dataset 'vkitti2'

python post_process/generate_statistic.py --logdir ./checkpoints/vkitti2/gwcnet-gc-sml1-lr1e-4/move/ \
    --epochs 0 \
    --maxdisp 192 \
    --inliers 0 \
    --mask 'soft' \
    --bin_scale 'log' \
    --n_bins 11 \
    --dataset 'vkitti2'

python post_process/generate_statistic.py --logdir ./checkpoints/vkitti2/gwcnet-gc-elu-dropout-l1-logs-lr1e-4/move/ \
    --epochs 0 \
    --maxdisp 192 \
    --inliers 0 \
    --mask 'hard' \
    --bin_scale 'log' \
    --n_bins 11 \
    --uncert \
    --dataset 'vkitti2'

python post_process/generate_statistic.py --logdir ./checkpoints/vkitti2/gwcnet-gc-elu-dropout-l1-logs-inliers-lr1e-4/move/ \
    --epochs 0 \
    --maxdisp 192 \
    --inliers 5 \
    --mask 'hard' \
    --bin_scale 'log' \
    --n_bins 11 \
    --uncert \
    --dataset 'vkitti2'

python post_process/generate_statistic.py --logdir ./checkpoints/vkitti2/gwcnet-gc-elu-dropout-l1-logs-kl-1std-lr1e-4/move/ \
    --epochs 0 \
    --maxdisp 192 \
    --inliers 1 \
    --mask 'soft' \
    --bin_scale 'log' \
    --n_bins 11 \
    --uncert \
    --dataset 'vkitti2'

python post_process/generate_statistic.py --logdir ./checkpoints/vkitti2/gwcnet-gc-elu-dropout-l1-logs-kl-3std-lr1e-4/move/ \
    --epochs 0 \
    --maxdisp 192 \
    --inliers 3 \
    --mask 'soft' \
    --bin_scale 'log' \
    --n_bins 11 \
    --uncert \
    --dataset 'vkitti2'

python post_process/generate_statistic.py --logdir ./checkpoints/vkitti2/gwcnet-gc-elu-dropout-l1-logs-kl-5std-logspace-lr1e-4/move/ \
    --epochs 0 \
    --maxdisp 192 \
    --inliers 5 \
    --mask 'soft' \
    --bin_scale 'log' \
    --n_bins 11 \
    --uncert \
    --dataset 'vkitti2'
