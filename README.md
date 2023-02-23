# β-rank: A Robust Convolutional Filter Pruning Method for Imbalanced Medical Image Analysis!

## Contents
- [Overview](#overview)
- [Before training](#before-training)
- [How to train](#how-to-train)
- [Trained models and log files](#trained-models-and-log-files)
- [Results](#results)
- [Android Demo](#android-demo)
- [Citation](#citation)

## Overview
The ranking of β-rank is based on the following equation where $R_{i_k}^{L1}=\sum|f_{i_k}|$ is absolute summation of a filter values.

<img src="https://github.com/mohofar/Beta-Rank/blob/main/images/Picture1.png" width=65% height=65%>

<img src="https://github.com/mohofar/Beta-Rank/blob/main/images/Equation5.png" width=55% height=55%>



## Before training
The folders structre are as follow:

<img src="https://github.com/mohofar/Beta-Rank/blob/main/images/F1.png" width=40% height=40%>
<img src="https://github.com/mohofar/Beta-Rank/blob/main/images/F2.png" width=40% height=40%>

Note: In order to make the repository smaller, we have excluded the data and baseline folders from our uploads. However, if you require the datasets for training purposes, you may obtain the folders from the links provided. Additionally, please download the [```data```](https://drive.google.com/drive/folders/1cY2FqpykVAO_M0qiyQsqQXeF8UOKxAzB?usp=sharing) and [```baseline```](https://drive.google.com/drive/folders/1zgYYArsM7p2UhkgaHfQarc8OUUlPDpJU?usp=sharing) folders from the provided links and include them with the others. Alternatively, you can create empty folders for them Lastly, note that the "rank_conv" folder is currently empty and will be used to store the training log.




## How to train
Example: Cifar10 and VGG16 model training using Beta-rank method

1. Generate ranking of filters using ```rank_generation.py``` by providing enough information like this sample:


```shell
!python rank_generation.py \
--pretrain_dir './baseline/Baseline_VGG_cifar10/vgg_16_bn.pt' \
--arch 'vgg_16_bn' \
--pruning_method 'L1'\
--dataset 'cifar10' \
--num_class 10 \
--batch_size 128
```


2. Use this example as a guide for training a pruned model using ```tuning.py```:


```shell
!python tuning.py \
--job_dir './rank_conv/vgg_16_bn_L1_cifar10_limit1' \
--arch 'vgg_16_bn' \
--use_pretrain \
--pretrain_dir './baseline/Baseline_VGG_cifar10/vgg_16_bn.pt' \
--compress_rate [0.65]*7+[0.8]*5 \
--rank_conv_prefix './rank_conv/vgg_16_bn_L1_cifar10_limit1' \
--num_class 10\
--dataset 'cifar10'\
--epochs 5
```

Note: To observe the outcomes of a sample run, refer to the ```Sample_running.ipynb``` file.

## Trained Models and Log Files
All the configurations, ranking of filters, training logs, and trained weights  that are mentioned in the paper can be found at this [link](https://drive.google.com/drive/folders/1ys83lqR5rhUaegPy1lZxHYLkOixhJ38l?usp=sharing).
## Results
The tables below present the results obtained by three methods under identical training conditions.
<img src="https://github.com/mohofar/Beta-Rank/blob/main/images/tt1.png" width=65% height=65%>
<img src="https://github.com/mohofar/Beta-Rank/blob/main/images/t2.png" width=65% height=65%>

In order to assess the stability through different experiments (E1 and E2) of various methods (M1 and M2), we have introduced the following equation, which considers filters in the form of colorful shapes.

<img src="https://github.com/mohofar/Beta-Rank/blob/main/images/Sa_St.png" width=25% height=25%>

The following plots show the stability of filter selection using different filter pruning methods (Blue: Beta-rank, Orange: Hrank) for ResNet56. The actual values are presented in gray color and smoothed versions of them are presented in blue and orange. The best value for fraction of stability is 0.25 in this analysis as we select 25% of the top and least ranked filters to explore.

<img src="https://github.com/mohofar/Beta-Rank/blob/main/images/St.png" width=55% height=55%>


## Android Demo (Under Reconstruction!!)

## Citation

## Note
We are grateful to the authors of the Hrank paper (https://github.com/lmbxmu/HRankPlus) for sharing their code, which we used as a basis for our own work.
