#!/usr/bin/env bash
python3 -m torch.distributed.launch --nproc_per_node=1 feature_extract.py \
 --model "vgg" \
 --path "Exp_base/vgg_base" \
 --dataset "cifar10" \
 --save_path 'your_data_path' \
 --target_flops 100000000 \
 --beta 231 \
 --test_batch_size 1024
