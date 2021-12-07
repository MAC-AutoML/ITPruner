#!/usr/bin/env bash
code_path='/ITPruner/Imagenet/'
chmod +x ${code_path}/prep_imagenet.sh
cd ${code_path}
echo "preparing data"
bash ${code_path}/prep_imagenet.sh >> /dev/null
echo "preparing data finished"
python3 -m torch.distributed.launch --nproc_per_node=4 train.py --train \
 --model "resnet50" \
 --cfg "[52, 55, 55, 146, 55, 55, 55, 55, 105, 111, 232, 111, 111, 111, 111, 111, 111, 211, 222, 290, 222, 222, 222, 222, 222, 222, 222, 222, 222, 222, 423, 442, 1453, 442, 442, 442, 442]" \
 --path "Exp_train/train_resnet50_2300m_${RANDOM}" \
 --dataset "imagenet" \
 --save_path 'your_data_path' \
 --base_path "Exp_base/resnet50_base" \
 --sync_bn \
 --warm_epoch 1 \
 --n_epochs 120
