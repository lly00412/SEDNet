# SEDNet

This is the Python implementation of the SEDNet with GwcNet backbone. (**Learning the Distribution of Errors in Stereo Matching for Joint Disparity and Uncertainty Estimation**. Published at CVPR 2023) [[Paper](https://arxiv.org/abs/2304.00152)]

## Environment
* python 3.7
* Pytorch >= 0.10.2
* Cuda >= 11.0
* Anaconda
* Create environment by `conda env create -f sednet.yml` or `conda create --name myenv --file sednet.txt`

## Data Preparation
Download datasets at:
* [Scene Flow Datasets](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
* [KITTI 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)
* [Virtual KITTI 2](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/)
* [DrivingStereo](https://drivingstereo-dataset.github.io/)

## Training

### Training List
* [Training/Test split](https://github.com/lly00412/SEDNet/tree/main/filenames) used in the paper.
* You can also use files in [generate_datas](https://github.com/lly00412/SEDNet/blob/main/generate_datas) to generate your own split.
* Please save the split files to `./filenames`.

### Training Scripts
* `main.py` is used to training the SEDNet.
* Training scripts are saved in `./scripts`
* For `--losstype`, `smooth_l1` is the smooth L1 loss in [Guo et al.](https://arxiv.org/pdf/1903.04025.pdf), `KG` is the log-likelihood loss in [Kendall and Gal.](https://proceedings.neurips.cc/paper/2017/file/2650d6089a6d640c5e85b2b88265dc2b-Paper.pdf), `UC` is our novel divergence loss with the log-likelihood loss.
* To train the LAF baseline, you need to run `./generate_datas/generate_laf_data.py` to save the cost column of stereo network at first.

**Example of Scene Flow Datasets**

* run the scripts `./scripts/sceneflow.sh` to traing on Scene Flow datasets
* Please update `DATAPATH` and `SAVEPATH` as your training data path and the log/checkpoints save path.
* You can use `--loadckpt` to specific the pre-trained checkpoint file.

## Evaluation

* Files in [post_process](https://github.com/lly00412/SEDNet/tree/main/post_process) are used to evaluate the models.
* `generate_statistic.py` is to compute the evaluation metrics.
* `generate_conf_and_depth.py` can covert the disparity maps and uncertainty maps to depth maps and the confidence maps via gaussian error function.
* Run `./scripts/sceneflow_analysis.sh` to generate the evaluation metric of models trained with Scene Flow datasets.

## Save Outputs

* Run `./scripts/kitti15_save.sh` to save the disparity maps of the model that fine-tunning on KIITI 2015 dataset. Please update the `--loadckpt` as your checkpoint file to generate the disparity maps.

## Pretrained Models
* [SceneFlow](https://github.com/lly00412/SEDNet/blob/c2be9da4e9d3e534fcc1f883f1d3a4be4b14516a/checkpoints/sceneflow/gwcnet-gc-elu-dropout-l1-logs-kl-3std-logspace-11bins-lr1e-3/checkpoint_000015.ckpt): SEDNet with a soft inlier threshold of 3 sigma and 11 bins in logspace.
* [VKITT2](https://github.com/lly00412/SEDNet/blob/c2be9da4e9d3e534fcc1f883f1d3a4be4b14516a/checkpoints/vkitti2/sednet-gwc-3std-lr1e-4/checkpoint_000025.ckpt): SEDNet with a soft inlier threshold of 3 sigma and 11 bins in logspace.

## Citation
'''
@article{chen2023learning,
  title={Learning the Distribution of Errors in Stereo Matching for Joint Disparity and Uncertainty Estimation},
  author={Chen, Liyan and Wang, Weihan and Mordohai, Philippos},
  journal={arXiv e-prints},
  pages={arXiv--2304},
  year={2023}
}
'''
