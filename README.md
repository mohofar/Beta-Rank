# Beta-Rank (Under Reconstruction!!)

β-rank: A Robust Convolutional Filter Pruning Method for Imbalanced Medical Image Analysis!

## Contents
- [Overview](#overview)
- [How to train](how-to-train)
- [Trained models and log files](#trained-models-and-log-files)
- [How to evaluate](#how-to-evaluate)
- [Results](#results)
- [Android Demo](#android-demo)
- [Citation](#citation)

## Overview
The presented method for pruning convolutional networks is presented in the following:

<img src="https://github.com/mohofar/Beta-Rank/blob/main/images/Picture1.png" width=100% height=100%>

The ranking of the our method is based on the following equation where $R_{i_k}^{L1}=\sum|f_{i_k}|$ is absolute summation of a filter values.


<img src="https://github.com/mohofar/Beta-Rank/blob/main/images/Equation5.png" width=65% height=65%>


## How to train
Example: Cifar10 and VGG16 model training using Beta-rank method

1. Generate ranking of filters using ```rank_generation_c10.py``` by providing enough information:

```shell
!python rank_generation_c10.py \
--pretrain_dir './baseline/Baseline_VGG_cifar10/vgg_16_bn.pt' \
--arch 'vgg_16_bn' \
--pruning_method 'Beta'\
--batch_size 16
```

2. Train pruned model 

```shell
!python tuning_cifar10.py \
--job_dir './rank_conv/vgg_16_bn_Beta_limit1' \
--arch 'vgg_16_bn' \
--use_pretrain \
--pretrain_dir './baseline/Baseline_VGG_cifar10/vgg_16_bn.pt' \
--compress_rate [0.65]*7+[0.8]*5 \
--rank_conv_prefix './rank_conv/vgg_16_bn_Beta_limit1' 
```
## Trained Models and Log Files

## How to Evaluate 

## Results

## Android Demo

## Citation

## Note
We used Hrank paper codes as our base codes to develop!
