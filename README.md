# Code for "Long-tailed learning with in- and out-of-distribution noisy labels in the open-world"
In this paper, we propose the Imbalanced Noisy Labels Calibration (INLC) approach.

## Preparation
Main dependency:
```txt
loguru             0.7.3
torch              2.5.1
```
The datasets used in the experiments: 
- ILSVRC2012: Images - Validation images (all tasks). 6.3GB. [[link]](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php)
- ImageNet32: Download downsampled image data (32x32, 64x64) - Train(32x32), 3 GB [[link]](https://image-net.org/download-images.php)
- WebVision1.0: Resized Images (small version) - Google Images Resized (16 GB) [[link]](https://data.vision.ee.ethz.ch/cvl/webvision//download.html)

We put the datasets in the `~/data` directory, and the directory structure is as follows:
```txt
├─cifar-10-batches-py
├─cifar-100-python
├─ILSVRC2012
│  ├─ILSVRC2012_devkit_t12
│  │  ├─data
│  │  └─evaluation
│  └─ILSVRC2012_img_val
├─imagenet32
│  ├─train_data_batch_1
│  ├─...
│  └─val_data
└─webvision1.0
    ├─google
    │  ├─q0001
    │  ├─q0002
    │  ├─...
    |  └─q1632
    ├─info
    └─val_images_256
```
## Usage
Training in CIFAR-10
```bash
python main_cifar.py
python main_cifar.py --r_ood 0.2
python main_cifar.py --r_ood 0.2 --r_id 0.2
python main_cifar.py --r_ood 0.2 --r_id 0.2 --asym
python main_cifar.py --r_imb 0.01
python main_cifar.py --r_imb 0.01 --r_ood 0.2
python main_cifar.py --r_imb 0.01 --r_ood 0.2 --r_id 0.2
python main_cifar.py --r_imb 0.01 --r_ood 0.2 --r_id 0.2 --asym
```
Try strategy 2 or replace loss with focal in CIFAR-10
```bash
python main_cifar.py --s2
python main_cifar.py --loss focal
```
Training in CIFAR-100
```bash
python main_cifar.py --dataset 100 --tau 0.6
python main_cifar.py --dataset 100 --tau 0.6 --r_ood 0.2
python main_cifar.py --dataset 100 --tau 0.6 --r_ood 0.2 --r_id 0.2 --warm_epochs 20
python main_cifar.py --dataset 100 --tau 0.6 --r_ood 0.2 --r_id 0.2 --asym --warm_epochs 20
python main_cifar.py --dataset 100 --tau 0.6 --r_imb 0.01
python main_cifar.py --dataset 100 --tau 0.6 --r_imb 0.01 --r_ood 0.2
python main_cifar.py --dataset 100 --tau 0.6 --r_imb 0.01 --r_ood 0.2 --r_id 0.2 --warm_epochs 20
python main_cifar.py --dataset 100 --tau 0.6 --r_imb 0.01 --r_ood 0.2 --r_id 0.2 --asym --warm_epochs 20
```
Training or evaluating in WebVision 1.0
```bash
python main_webvision.py --s2
python main_webvision.py --test
```
If training collapses, try increasing the value of `--warm_epochs`.

## Reference
- [Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss](https://github.com/kaidic/LDAM-DRW)
- [DivideMix: Learning with Noisy Labels as Semi-supervised Learning](https://github.com/LiJunnan1992/DivideMix)
- [DASO: Distribution-Aware Semantics-Oriented Pseudo-label for Imbalanced Semi-Supervised Learning](https://github.com/ytaek-oh/daso)
- [Jo-SRC: A Contrastive Approach for Combating Noisy Labels](https://github.com/NUST-Machine-Intelligence-Laboratory/Jo-SRC)
