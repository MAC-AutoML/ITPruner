#!/usr/bin/env bash
code_path='/ITPruner/Imagenet/'
chmod +x ${code_path}/prep_imagenet.sh
cd ${code_path}
echo "preparing data"
bash ${code_path}/prep_imagenet.sh >> /dev/null
echo "preparing data finished"
python3 -m torch.distributed.launch --nproc_per_node=4 train.py --train \
 --model "resnet50" \
 --cfg "[48, 52, 52, 110, 52, 52, 52, 52, 98, 105, 142, 105, 105, 105, 105, 105, 105, 197, 211, 48, 211, 211, 211, 211, 211, 211, 211, 211, 211, 211, 397, 420, 1264, 419, 419, 419, 418]" \
 --path "Exp_train/train_resnet50_1800m_${RANDOM}" \
 --dataset "imagenet" \
 --save_path 'your_data_path' \
 --sync_bn \
 --warm_epoch 1 \
 --base_path "Exp_base/resnet50_base" \
 --n_epochs 120
