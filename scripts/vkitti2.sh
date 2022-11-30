#!/usr/bin/env bash
set -e
VK2PATH="/media/Data/datasets/vkitti2/"
DSPATH="/media/Data/datasets/DrivingStereo/"
SAVEPATH="/mnt/Data2/liyan/GwcNet/"
SUFX=("morning" "sunset" "rain" "fog")
WEATHER=("sunny" "foggy" "cloudy" "rainy")

# GwcNet
LOADDIR="gwcnet-gc-sml1-lr1e-4"
CKPT="checkpoint_000020.ckpt"
echo $LOADDIR
echo $CKPT
for S in ${SUFX[@]}
do
  echo $S
  python main.py --dataset vkitti2 \
    --datapath $VK2PATH --trainlist ./filenames/vkitti2_train_wS6.txt --testlist ./filenames/vkitti2_test_$S.txt \
    --maxdisp 192 \
    --epochs 1 --lrepochs " 7,20,30,40,50:5" \
    --batch_size 8 \
    --lr 0.0001 \
    --loss_type 'smooth_l1' \
    --save_test \
    --mask 'soft' \
    --bin_scale 'log' \
    --n_bins 11 \
    --inliers 0 \
    --model gwcnet-gc \
    --loadckpt $SAVEPATH/checkpoints/vkitti2/$LOADDIR/$CKPT \
    --logdir $SAVEPATH/checkpoints/vkitti2/$LOADDIR/$S/ \
    --test_batch_size 1 \
    --device_id 0 1

  python post_process/generate_statistic.py --logdir $SAVEPATH/checkpoints/vkitti2/$LOADDIR/$S/ \
    --epochs 0 \
    --maxdisp 192 \
    --inliers 0 \
    --mask soft \
    --bin_scale log \
    --n_bins 11 \
    --dataset vkitti2
done

for W in ${WEATHER[@]}
do
  echo $W
  python main.py --dataset drivingstereo \
    --datapath $DSPATH --trainlist ./filenames/vkitti2_train_wS6.txt --testlist ./filenames/drivingstereo_test_$W.txt \
    --maxdisp 192 \
    --epochs 1 --lrepochs " 7,20,30,40,50:5" \
    --batch_size 8 \
    --lr 0.0001 \
    --loss_type 'smooth_l1' \
    --save_test \
    --mask 'soft' \
    --bin_scale 'log' \
    --n_bins 11 \
    --inliers 0 \
    --model gwcnet-gc \
    --loadckpt $SAVEPATH/checkpoints/vkitti2/$LOADDIR/$CKPT \
    --logdir $SAVEPATH/checkpoints/drivingstereo/$LOADDIR/$W/ \
    --test_batch_size 1 \
    --device_id 0 1

  python post_process/generate_statistic.py --logdir $SAVEPATH/checkpoints/drivingstereo/$LOADDIR/$W/ \
    --epochs 0 \
    --maxdisp 192 \
    --inliers 0 \
    --mask soft \
    --bin_scale log \
    --n_bins 11 \
    --dataset drivingstereo
done

# l_log
LOADDIR="gwcnet-gc-elu-dropout-l1-logs-lr1e-4"
CKPT="checkpoint_000029.ckpt"
echo $LOADDIR
echo $CKPT
for S in ${SUFX[@]}
do
  echo $S
  python main.py --dataset vkitti2 \
    --datapath $VK2PATH --trainlist ./filenames/vkitti2_train_wS6.txt --testlist ./filenames/vkitti2_test_$S.txt \
    --maxdisp 192 \
    --epochs 1 --lrepochs " 7,20,30,40,50:5" \
    --batch_size 8 \
    --lr 0.0001 \
    --loss_type 'KG' \
    --save_test \
    --mask 'hard' \
    --bin_scale 'log' \
    --n_bins 11 \
    --inliers 0 \
    --model gwcnet-gcs \
    --loadckpt $SAVEPATH/checkpoints/vkitti2/$LOADDIR/$CKPT \
    --logdir $SAVEPATH/checkpoints/vkitti2/$LOADDIR/$S/ \
    --test_batch_size 1 \
    --device_id 0 1

  python post_process/generate_statistic.py --logdir $SAVEPATH/checkpoints/vkitti2/$LOADDIR/$S/ \
    --epochs 0 \
    --maxdisp 192 \
    --inliers 0 \
    --mask soft \
    --bin_scale log \
    --n_bins 11 \
    --uncert \
    --dataset vkitti2
done

for W in ${WEATHER[@]}
do
  echo $W
  python main.py --dataset drivingstereo \
    --datapath $DSPATH --trainlist ./filenames/vkitti2_train_wS6.txt --testlist ./filenames/drivingstereo_test_$W.txt \
    --maxdisp 192 \
    --epochs 1 --lrepochs " 7,20,30,40,50:5" \
    --batch_size 8 \
    --lr 0.0001 \
    --loss_type 'KG' \
    --save_test \
    --mask 'hard' \
    --bin_scale 'log' \
    --n_bins 11 \
    --inliers 0 \
    --model gwcnet-gcs \
    --loadckpt $SAVEPATH/checkpoints/vkitti2/$LOADDIR/$CKPT \
    --logdir $SAVEPATH/checkpoints/drivingstereo/$LOADDIR/$W/ \
    --test_batch_size 1 \
    --device_id 0 1

  python post_process/generate_statistic.py --logdir $SAVEPATH/checkpoints/drivingstereo/$LOADDIR/$W/ \
    --epochs 0 \
    --maxdisp 192 \
    --inliers 0 \
    --mask soft \
    --bin_scale log \
    --n_bins 11 \
    --uncert \
    --dataset drivingstereo
done

# l_log epe<5
LOADDIR="gwcnet-gc-elu-dropout-l1-logs-inliers-lr1e-4"
CKPT="checkpoint_000018.ckpt"
echo $LOADDIR
echo $CKPT
for S in ${SUFX[@]}
do
  echo $S
  python main.py --dataset vkitti2 \
    --datapath $VK2PATH --trainlist ./filenames/vkitti2_train_wS6.txt --testlist ./filenames/vkitti2_test_$S.txt \
    --maxdisp 192 \
    --epochs 1 --lrepochs " 7,20,30,40,50:5" \
    --batch_size 8 \
    --lr 0.0001 \
    --loss_type 'KG' \
    --save_test \
    --mask 'hard' \
    --bin_scale 'log' \
    --n_bins 11 \
    --inliers 5 \
    --model gwcnet-gcs \
    --loadckpt $SAVEPATH/checkpoints/vkitti2/$LOADDIR/$CKPT \
    --logdir $SAVEPATH/checkpoints/vkitti2/$LOADDIR/$S/ \
    --test_batch_size 1 \
    --device_id 0 1

  python post_process/generate_statistic.py --logdir $SAVEPATH/checkpoints/vkitti2/$LOADDIR/$S/ \
    --epochs 0 \
    --maxdisp 192 \
    --inliers 5 \
    --mask hard \
    --bin_scale log \
    --n_bins 11 \
    --uncert \
    --dataset vkitti2
done

for W in ${WEATHER[@]}
do
  echo $W
  python main.py --dataset drivingstereo \
    --datapath $DSPATH --trainlist ./filenames/vkitti2_train_wS6.txt --testlist ./filenames/drivingstereo_test_$W.txt \
    --maxdisp 192 \
    --epochs 1 --lrepochs " 7,20,30,40,50:5" \
    --batch_size 8 \
    --lr 0.0001 \
    --loss_type 'KG' \
    --save_test \
    --mask 'hard' \
    --bin_scale 'log' \
    --n_bins 11 \
    --inliers 5 \
    --model gwcnet-gcs \
    --loadckpt $SAVEPATH/checkpoints/vkitti2/$LOADDIR/$CKPT \
    --logdir $SAVEPATH/checkpoints/drivingstereo/$LOADDIR/$W/ \
    --test_batch_size 1 \
    --device_id 0 1

  python post_process/generate_statistic.py --logdir $SAVEPATH/checkpoints/drivingstereo/$LOADDIR/$W/ \
    --epochs 0 \
    --maxdisp 192 \
    --inliers 5 \
    --mask hard \
    --bin_scale log \
    --n_bins 11 \
    --uncert \
    --dataset drivingstereo
done

# l_log epe<3std
LOADDIR="gwcnet-gcs-kg-11bins-3std-lr1e-4"
CKPT="checkpoint_000020.ckpt"
echo $LOADDIR
echo $CKPT
for S in ${SUFX[@]}
do
  echo $S
  python main.py --dataset vkitti2 \
    --datapath $VK2PATH --trainlist ./filenames/vkitti2_train_wS6.txt --testlist ./filenames/vkitti2_test_$S.txt \
    --maxdisp 192 \
    --epochs 1 --lrepochs " 7,20,30,40,50:5" \
    --batch_size 8 \
    --lr 0.0001 \
    --loss_type 'KG' \
    --save_test \
    --mask 'soft' \
    --bin_scale 'log' \
    --n_bins 11 \
    --inliers 3 \
    --model gwcnet-gcs \
    --loadckpt $SAVEPATH/checkpoints/vkitti2/$LOADDIR/$CKPT \
    --logdir $SAVEPATH/checkpoints/vkitti2/$LOADDIR/$S/ \
    --test_batch_size 1 \
    --device_id 0 1

  python post_process/generate_statistic.py --logdir $SAVEPATH/checkpoints/vkitti2/$LOADDIR/$S/ \
    --epochs 0 \
    --maxdisp 192 \
    --inliers 3 \
    --mask soft \
    --bin_scale log \
    --n_bins 11 \
    --uncert \
    --dataset vkitti2
done

for W in ${WEATHER[@]}
do
  echo $W
  python main.py --dataset drivingstereo \
    --datapath $DSPATH --trainlist ./filenames/vkitti2_train_wS6.txt --testlist ./filenames/drivingstereo_test_$W.txt \
    --maxdisp 192 \
    --epochs 1 --lrepochs " 7,20,30,40,50:5" \
    --batch_size 8 \
    --lr 0.0001 \
    --loss_type 'KG' \
    --save_test \
    --mask 'soft' \
    --bin_scale 'log' \
    --n_bins 11 \
    --inliers 3 \
    --model gwcnet-gcs \
    --loadckpt $SAVEPATH/checkpoints/vkitti2/$LOADDIR/$CKPT \
    --logdir $SAVEPATH/checkpoints/drivingstereo/$LOADDIR/$W/ \
    --test_batch_size 1 \
    --device_id 0 1

  python post_process/generate_statistic.py --logdir $SAVEPATH/checkpoints/drivingstereo/$LOADDIR/$W/ \
    --epochs 0 \
    --maxdisp 192 \
    --inliers 3 \
    --mask soft \
    --bin_scale log \
    --n_bins 11 \
    --uncert \
    --dataset drivingstereo
done

# sednet epe<1std
LOADDIR="gwcnet-gc-elu-dropout-l1-logs-kl-1std-lr1e-4"
CKPT="checkpoint_000059.ckpt"
echo $LOADDIR
echo $CKPT
for S in ${SUFX[@]}
do
  echo $S
  python main.py --dataset vkitti2 \
    --datapath $VK2PATH --trainlist ./filenames/vkitti2_train_wS6.txt --testlist ./filenames/vkitti2_test_$S.txt \
    --maxdisp 192 \
    --epochs 1 --lrepochs " 7,20,30,40,50:5" \
    --batch_size 8 \
    --lr 0.0001 \
    --loss_type 'UC' \
    --save_test \
    --mask 'soft' \
    --bin_scale 'log' \
    --n_bins 11 \
    --inliers 1 \
    --model gwcnet-gcs \
    --loadckpt $SAVEPATH/checkpoints/vkitti2/$LOADDIR/$CKPT \
    --logdir $SAVEPATH/checkpoints/vkitti2/$LOADDIR/$S/ \
    --test_batch_size 1 \
    --device_id 0 1

  python post_process/generate_statistic.py --logdir $SAVEPATH/checkpoints/vkitti2/$LOADDIR/$S/ \
    --epochs 0 \
    --maxdisp 192 \
    --inliers 1 \
    --mask soft \
    --bin_scale log \
    --n_bins 11 \
    --uncert \
    --dataset vkitti2
done

for W in ${WEATHER[@]}
do
  echo $W
  python main.py --dataset drivingstereo \
    --datapath $DSPATH --trainlist ./filenames/vkitti2_train_wS6.txt --testlist ./filenames/drivingstereo_test_$W.txt \
    --maxdisp 192 \
    --epochs 1 --lrepochs " 7,20,30,40,50:5" \
    --batch_size 8 \
    --lr 0.0001 \
    --loss_type 'UC' \
    --save_test \
    --mask 'soft' \
    --bin_scale 'log' \
    --n_bins 11 \
    --inliers 1 \
    --model gwcnet-gcs \
    --loadckpt $SAVEPATH/checkpoints/vkitti2/$LOADDIR/$CKPT \
    --logdir $SAVEPATH/checkpoints/drivingstereo/$LOADDIR/$W/ \
    --test_batch_size 1 \
    --device_id 0 1

  python post_process/generate_statistic.py --logdir $SAVEPATH/checkpoints/drivingstereo/$LOADDIR/$W/ \
    --epochs 0 \
    --maxdisp 192 \
    --inliers 1 \
    --mask soft \
    --bin_scale log \
    --n_bins 11 \
    --uncert \
    --dataset drivingstereo
done

# sednet epe<3std
LOADDIR="gwcnet-gc-elu-dropout-l1-logs-kl-3std-lr1e-4"
CKPT="checkpoint_000025.ckpt"
echo $LOADDIR
echo $CKPT
for S in ${SUFX[@]}
do
  echo $S
  python main.py --dataset vkitti2 \
    --datapath $VK2PATH --trainlist ./filenames/vkitti2_train_wS6.txt --testlist ./filenames/vkitti2_test_$S.txt \
    --maxdisp 192 \
    --epochs 1 --lrepochs " 7,20,30,40,50:5" \
    --batch_size 8 \
    --lr 0.0001 \
    --loss_type 'UC' \
    --save_test \
    --mask 'soft' \
    --bin_scale 'log' \
    --n_bins 11 \
    --inliers 3 \
    --model gwcnet-gcs \
    --loadckpt $SAVEPATH/checkpoints/vkitti2/$LOADDIR/$CKPT \
    --logdir $SAVEPATH/checkpoints/vkitti2/$LOADDIR/$S/ \
    --test_batch_size 1 \
    --device_id 0 1

  python post_process/generate_statistic.py --logdir $SAVEPATH/checkpoints/vkitti2/$LOADDIR/$S/ \
    --epochs 0 \
    --maxdisp 192 \
    --inliers 3 \
    --mask soft \
    --bin_scale log \
    --n_bins 11 \
    --uncert \
    --dataset vkitti2
done

for W in ${WEATHER[@]}
do
  echo $W
  python main.py --dataset drivingstereo \
    --datapath $DSPATH --trainlist ./filenames/vkitti2_train_wS6.txt --testlist ./filenames/drivingstereo_test_$W.txt \
    --maxdisp 192 \
    --epochs 1 --lrepochs " 7,20,30,40,50:5" \
    --batch_size 8 \
    --lr 0.0001 \
    --loss_type 'UC' \
    --save_test \
    --mask 'soft' \
    --bin_scale 'log' \
    --n_bins 11 \
    --inliers 3 \
    --model gwcnet-gcs \
    --loadckpt $SAVEPATH/checkpoints/vkitti2/$LOADDIR/$CKPT \
    --logdir $SAVEPATH/checkpoints/drivingstereo/$LOADDIR/$W/ \
    --test_batch_size 1 \
    --device_id 0 1

  python post_process/generate_statistic.py --logdir $SAVEPATH/checkpoints/drivingstereo/$LOADDIR/$W/ \
    --epochs 0 \
    --maxdisp 192 \
    --inliers 3 \
    --mask soft \
    --bin_scale log \
    --n_bins 11 \
    --uncert \
    --dataset drivingstereo
done

# sednet epe<5std
LOADDIR="gwcnet-gc-elu-dropout-l1-logs-kl-5std-logspace-lr1e-4"
CKPT="checkpoint_000031.ckpt"
echo $LOADDIR
echo $CKPT
for S in ${SUFX[@]}
do
  echo $S
  python main.py --dataset vkitti2 \
    --datapath $VK2PATH --trainlist ./filenames/vkitti2_train_wS6.txt --testlist ./filenames/vkitti2_test_$S.txt \
    --maxdisp 192 \
    --epochs 1 --lrepochs " 7,20,30,40,50:5" \
    --batch_size 8 \
    --lr 0.0001 \
    --loss_type 'UC' \
    --save_test \
    --mask 'soft' \
    --bin_scale 'log' \
    --n_bins 11 \
    --inliers 5 \
    --model gwcnet-gcs \
    --loadckpt $SAVEPATH/checkpoints/vkitti2/$LOADDIR/$CKPT \
    --logdir $SAVEPATH/checkpoints/vkitti2/$LOADDIR/$S/ \
    --test_batch_size 1 \
    --device_id 0 1

  python post_process/generate_statistic.py --logdir $SAVEPATH/checkpoints/vkitti2/$LOADDIR/$S/ \
    --epochs 0 \
    --maxdisp 192 \
    --inliers 5 \
    --mask soft \
    --bin_scale log \
    --n_bins 11 \
    --uncert \
    --dataset vkitti2
done

for W in ${WEATHER[@]}
do
  echo $W
  python main.py --dataset drivingstereo \
    --datapath $DSPATH --trainlist ./filenames/vkitti2_train_wS6.txt --testlist ./filenames/drivingstereo_test_$W.txt \
    --maxdisp 192 \
    --epochs 1 --lrepochs " 7,20,30,40,50:5" \
    --batch_size 8 \
    --lr 0.0001 \
    --loss_type 'UC' \
    --save_test \
    --mask 'soft' \
    --bin_scale 'log' \
    --n_bins 11 \
    --inliers 5 \
    --model gwcnet-gcs \
    --loadckpt $SAVEPATH/checkpoints/vkitti2/$LOADDIR/$CKPT \
    --logdir $SAVEPATH/checkpoints/drivingstereo/$LOADDIR/$W/ \
    --test_batch_size 1 \
    --device_id 0 1

  python post_process/generate_statistic.py --logdir $SAVEPATH/checkpoints/drivingstereo/$LOADDIR/$W/ \
    --epochs 0 \
    --maxdisp 192 \
    --inliers 5 \
    --mask soft \
    --bin_scale log \
    --n_bins 11 \
    --uncert \
    --dataset drivingstereo
done