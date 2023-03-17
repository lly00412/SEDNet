#!/usr/bin/env bash
set -e
VK2PATH="/media/Data2/liyan/datasets/vikitti2/"
SAVEPATH="/media/Data2/liyan/checkpoints/vkitti2/"

#python main.py --dataset vkitti2 \
#    --datapath $VK2PATH --trainlist ./filenames/vkitti2_train.txt --testlist ./filenames/vkitti2_test.txt \
#    --maxdisp 192 \
#    --epochs 1 --lrepochs " 7,20,30,40,50:5" \
#    --batch_size 16 \
#    --lr 0.0001 \
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
#    --device_id 0 1
#
#python main.py --dataset vkitti2 \
#    --datapath $VK2PATH --trainlist ./filenames/vkitti2_train.txt --testlist ./filenames/vkitti2_test.txt \
#    --maxdisp 192 \
#    --epochs 1 --lrepochs " 7,20,30,40,50:5" \
#    --batch_size 16 \
#    --lr 0.0001 \
#    --loss_type 'smooth_l1' \
#    --training \
#    --save_test \
#    --mask 'hard' \
#    --inliers 5 \
#    --model gwcnet-gc \
#    --logdir $SAVEPATH/gwc-gc-sml1-epe5/ \
#    --test_batch_size 2 \
#    --device_id 0 1


python post_process/generate_statistic.py --logdir /mnt/Data2/liyan/GwcNet/checkpoints/drivingstereo/gwcnet-gcs-uc-5std-log-11b-vkpt-lr1e-4/weather/ \
    --epochs 0 \
    --maxdisp 192 \
    --inliers 5 \
    --mask 'soft' \
    --bin_scale 'log' \
    --n_bins 11 \
    --uncert \
    --dataset 'drivingstereo'