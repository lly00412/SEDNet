#!/usr/bin/env bash
set -e
DATAPATH="../datasets/SceneFlow/LAF/gwcnet-gc/"
#python train.py --datapath $DATAPATH \
#      --dataset sceneflow \
#      --logdir ./LAF/checkpoints/scenflow/gwcnet-gc/ \
#      --disp_model gwcnet-gc \
#      --maxdisp 192 \
#      --zoom 0.5 \
#      --resume \
#      --crop_w 448 \
#      --crop_h 284 \
#      --batch_size 8 \
#      --base_lr 3e-4 \
#      --test_batch_size 1 \
#      --save_test

python post_process/generate_statistic.py --logdir ./LAF/checkpoints/scenflow/gwcnet-gc/ \
    --epochs 16 \
    --inliers 0 \
    --bin_scale 'log' \
    --n_bins 11



# killed by 012698,031355,013277,00793