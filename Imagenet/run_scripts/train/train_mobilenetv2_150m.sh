#!/usr/bin/env bash
code_path='/ITPruner/Imagenet/'
chmod +x ${code_path}/prep_imagenet.sh
cd ${code_path}
echo "preparing data"
bash ${code_path}/prep_imagenet.sh >> /dev/null
echo "preparing data finished"
python3 -m torch.distributed.launch --nproc_per_node=4 train.py --train \
 --model "mobilenetv2" \
 --cfg "[29, 12, 16, 24, 43, 50, 98, 273]" \
 --path "Exp_train/train_mobilenetv2_150m_${RANDOM}" \
 --dataset "imagenet" \
 --save_path 'your_data_path' \
 --base_path "Exp_base/mobilenetv2_base" \
 --warm_epoch 1 \
 --sync_bn \
 --n_epochs 250 \
 --label_smoothing 0.1
