#!/usr/bin/env bash
python3 -m torch.distributed.launch --nproc_per_node=1 evaluate.py \
 --model "resnet50" \
 --path "pretrain_models/train_resnet50_2300m_12067" \
 --dataset "imagenet" \
 --save_path 'your_data_path' \
 --cfg "[52, 55, 55, 146, 55, 55, 55, 55, 105, 111, 232, 111, 111, 111, 111, 111, 111, 211, 222, 290, 222, 222, 222, 222, 222, 222, 222, 222, 222, 222, 423, 442, 1453, 442, 442, 442, 442]"