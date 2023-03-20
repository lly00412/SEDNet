#!/usr/bin/env bash
set -e
VK2PATH="/media/Data/datasets/vkitti2/"
DSPATH="/media/Data/datasets/DrivingStereo/"
SAVEPATH="/mnt/Data2/liyan/GwcNet/checkpoints/"
SUFX=("move" "morning" "sunset" "rain" "fog")
WEATHER=("sunny" "foggy" "cloudy" "rainy")

####################################
#          GwcNet+epe5
####################################
LOADDIR="gwc-gc-sml1-epe5"
CKPT="checkpoint_000049.ckpt"

echo $LOADDIR
echo $CKPT

#evaluate for S6
python post_process/generate_statistic.py --logdir $SAVEPATH/vkitti2/$LOADDIR/ \
    --epochs 49 \
    --maxdisp 192 \
    --inliers 5 \
    --mask hard \
    --dataset vkitti2

#evaluate for S6 subset
for S in ${SUFX[@]}
do
  echo $S
  python main.py --dataset vkitti2 \
    --datapath $VK2PATH --trainlist ./filenames/vkitti2_train.txt --testlist ./filenames/vkitti2_test_$S.txt \
    --maxdisp 192 \
    --epochs 1 --lrepochs " 7,20,30,40,50:5" \
    --batch_size 16 \
    --lr 0.0001 \
    --loss_type 'smooth_l1' \
    --save_test \
    --mask 'hard' \
    --inliers 5 \
    --model gwcnet-gc \
    --loadckpt $SAVEPATH/vkitti2/$LOADDIR/$CKPT \
    --logdir $SAVEPATH/vkitti2/$LOADDIR/$S/ \
    --test_batch_size 2 \
    --device_id 0 1

  python post_process/generate_statistic.py --logdir $SAVEPATH/vkitti2/$LOADDIR/$S/ \
    --epochs 0 \
    --maxdisp 192 \
    --inliers 5 \
    --mask hard \
    --dataset vkitti2
done

for W in ${WEATHER[@]}
do
  echo $W
  python main.py --dataset drivingstereo \
    --datapath $DSPATH --trainlist ./filenames/vkitti2_train.txt --testlist ./filenames/drivingstereo_test_$W.txt \
    --maxdisp 192 \
    --epochs 1 --lrepochs " 7,20,30,40,50:5" \
    --batch_size 16 \
    --lr 0.0001 \
    --loss_type 'smooth_l1' \
    --save_test \
    --mask 'hard' \
    --inliers 5 \
    --model gwcnet-gc \
    --loadckpt $SAVEPATH/vkitti2/$LOADDIR/$CKPT \
    --logdir $SAVEPATH/drivingstereo/$LOADDIR/$W/ \
    --test_batch_size 2 \
    --device_id 0 1

  python post_process/generate_statistic.py --logdir $SAVEPATH/drivingstereo/$LOADDIR/$W/ \
    --epochs 0 \
    --maxdisp 192 \
    --inliers 5 \
    --mask hard \
    --dataset drivingstereo
done

####################################
#          GwcNet+3std
####################################
LOADDIR="gwc-gc-sml1-log-3std"
CKPT="checkpoint_000049.ckpt"

echo $LOADDIR
echo $CKPT

#evaluate for S6
python post_process/generate_statistic.py --logdir $SAVEPATH/vkitti2/$LOADDIR/ \
    --epochs 49 \
    --maxdisp 192 \
    --inliers 3 \
    --mask soft \
    --bin_scale log \
    --n_bins 11 \
    --dataset vkitti2

#evaluate for S6 subset
for S in ${SUFX[@]}
do
  echo $S
  python main.py --dataset vkitti2 \
    --datapath $VK2PATH --trainlist ./filenames/vkitti2_train.txt --testlist ./filenames/vkitti2_test_$S.txt \
    --maxdisp 192 \
    --epochs 1 --lrepochs " 7,20,30,40,50:5" \
    --batch_size 16 \
    --lr 0.0001 \
    --loss_type 'smooth_l1' \
    --save_test \
    --inliers 3 \
    --mask soft \
    --bin_scale log \
    --n_bins 11 \
    --model gwcnet-gc \
    --loadckpt $SAVEPATH/vkitti2/$LOADDIR/$CKPT \
    --logdir $SAVEPATH/vkitti2/$LOADDIR/$S/ \
    --test_batch_size 2 \
    --device_id 0 1

  python post_process/generate_statistic.py --logdir $SAVEPATH/vkitti2/$LOADDIR/$S/ \
    --epochs 0 \
    --maxdisp 192 \
    --inliers 3 \
    --mask soft \
    --bin_scale log \
    --n_bins 11 \
    --dataset vkitti2
done

for W in ${WEATHER[@]}
do
  echo $W
  python main.py --dataset drivingstereo \
    --datapath $DSPATH --trainlist ./filenames/vkitti2_train.txt --testlist ./filenames/drivingstereo_test_$W.txt \
    --maxdisp 192 \
    --epochs 1 --lrepochs " 7,20,30,40,50:5" \
    --batch_size 16 \
    --lr 0.0001 \
    --loss_type 'smooth_l1' \
    --save_test \
    --inliers 3 \
    --mask soft \
    --bin_scale log \
    --n_bins 11 \
    --model gwcnet-gc \
    --loadckpt $SAVEPATH/vkitti2/$LOADDIR/$CKPT \
    --logdir $SAVEPATH/drivingstereo/$LOADDIR/$W/ \
    --test_batch_size 2 \
    --device_id 0 1

  python post_process/generate_statistic.py --logdir $SAVEPATH/drivingstereo/$LOADDIR/$W/ \
    --epochs 0 \
    --maxdisp 192 \
    --inliers 3 \
    --mask soft \
    --bin_scale log \
    --n_bins 11 \
    --dataset drivingstereo
done

####################################
#          UC+epe5
####################################
LOADDIR="gwc-gcs-uc-epe5"
CKPT="checkpoint_000049.ckpt"

echo $LOADDIR
echo $CKPT

#evaluate for S6
python post_process/generate_statistic.py --logdir $SAVEPATH/vkitti2/$LOADDIR/ \
    --epochs 49 \
    --maxdisp 192 \
    --uncert \
    --inliers 5 \
    --mask hard \
    --dataset vkitti2

#evaluate for S6 subset
for S in ${SUFX[@]}
do
  echo $S
  python main.py --dataset vkitti2 \
    --datapath $VK2PATH --trainlist ./filenames/vkitti2_train.txt --testlist ./filenames/vkitti2_test_$S.txt \
    --maxdisp 192 \
    --epochs 1 --lrepochs " 7,20,30,40,50:5" \
    --batch_size 16 \
    --lr 0.0001 \
    --loss_type 'UC' \
    --save_test \
    --mask 'hard' \
    --inliers 5 \
    --model gwcnet-gcs \
    --loadckpt $SAVEPATH/vkitti2/$LOADDIR/$CKPT \
    --logdir $SAVEPATH/vkitti2/$LOADDIR/$S/ \
    --test_batch_size 2 \
    --device_id 0 1

  python post_process/generate_statistic.py --logdir $SAVEPATH/vkitti2/$LOADDIR/$S/ \
    --epochs 0 \
    --maxdisp 192 \
    --uncert \
    --inliers 5 \
    --mask hard \
    --dataset vkitti2
done

for W in ${WEATHER[@]}
do
  echo $W
  python main.py --dataset drivingstereo \
    --datapath $DSPATH --trainlist ./filenames/vkitti2_train.txt --testlist ./filenames/drivingstereo_test_$W.txt \
    --maxdisp 192 \
    --epochs 1 --lrepochs " 7,20,30,40,50:5" \
    --batch_size 16 \
    --lr 0.0001 \
    --loss_type 'UC' \
    --save_test \
    --mask 'hard' \
    --inliers 5 \
    --model gwcnet-gcs \
    --loadckpt $SAVEPATH/vkitti2/$LOADDIR/$CKPT \
    --logdir $SAVEPATH/drivingstereo/$LOADDIR/$W/ \
    --test_batch_size 2 \
    --device_id 0 1

  python post_process/generate_statistic.py --logdir $SAVEPATH/drivingstereo/$LOADDIR/$W/ \
    --epochs 0 \
    --maxdisp 192 \
    --uncert \
    --inliers 5 \
    --mask hard \
    --dataset drivingstereo
done