# SEDNet

This is the official implementation of the paper **Learning the Distribution of Errors in Stereo Matching for Joint Disparity and Uncertainty Estimation**. (Submitted to CVPR 2023)

## Environment
* Anaconda
* GPU >= 1
* Create environment by `conda env create -f sednet.yml` or `conda create --name myenv --file sednet.txt`

## Data Preparation
Download datasets at:
* [Scene Flow Datasets](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
* [KITTI 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)
* [Virtual KITTI 2](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/)
* [DrivingStereo](https://drivingstereo-dataset.github.io/)

## Training

# Training List

# Training Scripts
* run the script `./scripts/vkitti2.sh`

* Please update `DATAPATH` and `--loadckpt` as your training data path and pretrained SceneFlow checkpoint file.

## Evaluation

## Save Outputs

## Pretrained Models

# Citation
