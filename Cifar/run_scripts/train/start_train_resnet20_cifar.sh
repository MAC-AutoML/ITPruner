## ============================= resnet cifar10 decode  ============================
python main.py \
  --gpu_id 0 \
  --exec_mode train \
  --learner vanilla \
  --dataset cifar10 \
  --data_path ~/datasets \
  --model_type resnet_decode \
  --lr 0.1 \
  --lr_min 0. \
  --lr_decy_type cosine \
  --weight_decay 5e-4 \
  --nesterov \
  --epochs 300 \
  --cfg 16,11,11,11,11,11,11,22,22,21,21,22,22,45,45,43,43,48,48

