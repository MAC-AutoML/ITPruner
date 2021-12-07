#!/usr/bin/env bash
python3 -m torch.distributed.launch --nproc_per_node=1 evaluate.py \
 --model "mobilenet" \
 --path "pretrain_models/train_mobilenet_285m_3956" \
 --dataset "imagenet" \
 --save_path 'your_data_path' \
 --cfg "[28, 52, 88, 90, 179, 182, 362, 311, 314, 314, 312, 367, 733, 1024]"