#!/usr/bin/env bash
set -e
#!/usr/bin/env bash
set -e
DATAPATH="/media/Data/datasets/vkitti2/"
python generate_laf_data.py --dataset vkitti2 \
    --datapath $DATAPATH --trainlist ./filenames/vkitti2_train_wS6.txt --testlist ./filenames/vkitti2_train_wS6.txt \
    --zoom 1.0 \
    --maxdisp 192 \
    --epochs 1 --lrepochs " 7,20,30,40,50:5" \
    --batch_size 8 \
    --lr 0.0001 \
    --training \
    --inliers 0 \
    --loss_type 'smooth_l1' \
    --save_test \
    --model gwcnet-gc \
    --loadckpt ./checkpoints/vkitti2/gwcnet-gc-sml1-lr1e-4/checkpoint_000020.ckpt \
    --logdir /mnt/Data2/datasets/vkitti2/LAF/gwcnet-gc/train \
    --test_batch_size 1

python generate_laf_data.py --dataset vkitti2 \
    --datapath $DATAPATH --trainlist ./filenames/vkitti2_test_wS6.txt --testlist ./filenames/vkitti2_test_wS6.txt \
    --zoom 1.0 \
    --maxdisp 192 \
    --epochs 1 --lrepochs " 7,20,30,40,50:5" \
    --batch_size 8 \
    --lr 0.0001 \
    --training \
    --inliers 0 \
    --loss_type 'smooth_l1' \
    --save_test \
    --model gwcnet-gc \
    --loadckpt ./checkpoints/vkitti2/gwcnet-gc-sml1-lr1e-4/checkpoint_000020.ckpt \
    --logdir /mnt/Data2/datasets/vkitti2/LAF/gwcnet-gc/test \
    --test_batch_size 1
