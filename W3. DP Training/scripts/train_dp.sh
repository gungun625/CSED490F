#!/bin/bash

CUDA_VISIBLE_DEVICES=0 nsys profile --force-overwrite=true --output nsight_logs/dp_gpu1 --trace=cuda,cublas,cudnn,nvtx,osrt python train_cifar.py --dp --data dataset/cifar10 --ckpt runs/default
CUDA_VISIBLE_DEVICES=0,1 nsys profile --force-overwrite=true --output nsight_logs/dp_gpu2 --trace=cuda,cublas,cudnn,nvtx,osrt python train_cifar.py --dp --data dataset/cifar10 --ckpt runs/dp_2