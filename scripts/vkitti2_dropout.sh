#!/usr/bin/env bash
set -x
DATAPATH="/media/Data2/liyan/datasets/vikitti2/"
SAVEPATH="/media/Data2/liyan/SEDNet/checkpoints/vkitti2-dropout/gwc-gc-l1/"

#python main.py --dataset vkitti2 \
#    --datapath $DATAPATH --trainlist ./filenames/vkitti2_train.txt --testlist ./filenames/vkitti2_test.txt \
#    --epochs 300 --lrepochs "150,300:5" \
#    --maxdisp 192 \
#    --model gwcnet-gc \
#    --batch_size 8 \
#    --training \
#    --lr 0.001 \
#    --loss_type 'smooth_l1' \
#    --save_test \
#    --bin_scale 'log' \
#    --n_bins 11 \
#    --inliers 5 \
#    --logdir $SAVEPATH/checkpoints/vkitti12/gwcnet-gc-l1-nodrop/ \
#    --test_batch_size 2

#python main.py --dataset vkitti2 \
#    --datapath $DATAPATH --trainlist ./filenames/vkitti2_train.txt --testlist ./filenames/vkitti2_test.txt \
#    --epochs 80 --lrepochs "150,300:5" \
#    --maxdisp 192 \
#    --model gwcnet-gc \
#    --batch_size 8 \
#    --training \
#    --lr 0.001 \
#    --loss_type 'smooth_l1' \
#    --save_test \
#    --bin_scale 'log' \
#    --n_bins 11 \
#    --inliers 5 \
#    --logdir $SAVEPATH/checkpoints/vkitti12/gwcnet-gc-l1-dropout-debug/ \
#    --test_batch_size 2

python main.py --dataset vkitti2 \
    --datapath $DATAPATH --trainlist ./filenames/vkitti2_train.txt --testlist ./filenames/vkitti2_test.txt \
    --epochs 80 --lrepochs "150,300:5" \
    --maxdisp 192 \
    --model gwcnet-gc \
    --batch_size 16 \
    --training \
    --lr 0.001 \
    --loss_type 'smooth_l1' \
    --save_test \
    --n_cluster 16 \
    --dropout 'dropout' \
    --drop_prob 0.5 \
    --device_id 0 1 \
    --logdir $SAVEPATH/drop_prob0.5/dropout/ \
    --test_batch_size 2