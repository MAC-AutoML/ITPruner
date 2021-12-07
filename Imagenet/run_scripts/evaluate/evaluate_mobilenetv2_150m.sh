#!/usr/bin/env bash
python3 -m torch.distributed.launch --nproc_per_node=1 evaluate.py \
 --model "mobilenetv2" \
 --path "pretrain_models/train_mobilenetv2_150m_19509" \
 --dataset "imagenet" \
 --save_path 'your_data_path' \
 --cfg "[29, 12, 16, 24, 43, 50, 98, 273]"