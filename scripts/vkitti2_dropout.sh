#!/usr/bin/env bash
set -x
DATAPATH="/media/Data/datasets/vkitti2/"
SAVEPATH="/mnt/Data2/liyan/GwcNet/checkpoints/vkitti2-dropout/gwc-gc-l1/"

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
    --batch_size 8 \
    --resume \
    --training \
    --lr 0.001 \
    --loss_type 'smooth_l1' \
    --save_test \
    --n_cluster 16 \
    --compute_cluster \
    --dropout 'dropcluster' \
    --drop_prob 0.5 \
    --device_id 0 1 \
    --maskdir $SAVEPATH/drop_prob0.5/dropcluster/clusters/ \
    --logdir $SAVEPATH/drop_prob0.5/dropcluster/ \
    --test_batch_size 2