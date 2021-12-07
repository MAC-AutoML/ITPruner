## ============================= resnet cifar10 decode  ============================
python main.py \
  --gpu_id 0 \
  --exec_mode eval \
  --learner vanilla \
  --dataset cifar10 \
  --data_path ~/datasets \
  --model_type resnet_decode \
  --load_path /ITPruner/Cifar/models/best_models/resnet20_20m/model.pt \
  --cfg 16,11,11,11,11,11,11,22,22,21,21,22,22,45,45,43,43,48,48

