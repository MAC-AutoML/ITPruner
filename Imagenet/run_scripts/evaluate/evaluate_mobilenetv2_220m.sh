#!/usr/bin/env bash
python3 -m torch.distributed.launch --nproc_per_node=1 evaluate.py \
 --model "mobilenetv2" \
 --path "pretrain_models/train_mobilenetv2_220m_19738" \
 --dataset "imagenet" \
 --save_path 'your_data_path' \
 --cfg "[31, 14, 20, 28, 54, 74, 130, 298]"