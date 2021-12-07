#!/usr/bin/env bash

python3 -m torch.distributed.launch --nproc_per_node=1 evaluate.py \
 --model "vgg" \
 --path "pretrain_models/train_vgg_100m_22031" \
 --dataset "cifar10" \
 --save_path 'your_data_path' \
 --cfg "[44, 35, 71, 71, 145, 104, 145, 290, 205, 325, 432, 429, 469]"
