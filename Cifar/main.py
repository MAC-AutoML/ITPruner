import traceback
import argparse
import torch
import torch.backends.cudnn as cudnn
from nets.model_helper import ModelHelper
from learners.basic_learner import BasicLearner
from data_loader import *
from utils.helper import *
import torch.multiprocessing as mp
import torch.distributed as dist
from utils.dist_utils import *
import os
import numpy as np
import random

parser = argparse.ArgumentParser()
parser.add_argument('--fix_random', action='store_true', help='fix randomness')
parser.add_argument('--seed', default=1234, type=int)
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--job_id', default='', help='generated automatically')
parser.add_argument('--exec_mode', default='', choices=['train', 'eval', 'finetune', 'misc'])
parser.add_argument('--print_freq', default=30, type=int, help='training printing frequency')
parser.add_argument('--model_type', default='', help='what model do you want to use')
parser.add_argument('--data_path', default='TODO', help='path to the dataset')
parser.add_argument('--data_aug', action='store_false', help='whether to use data augmentation')
parser.add_argument('--save_path', default='./models', help='save path for models')
parser.add_argument('--load_path', default='', help='path to load the model')
parser.add_argument('--load_path_log', default='record.log', help='suffix to name record.log')
parser.add_argument('--eval_epoch', type=int, default=1, help='perform eval at intervals')

# multi-process gpu
parser.add_argument('--dist-url', default='tcp://127.0.0.1', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--num_workers', type=int, default=4, help='num of workers data loaders')
parser.add_argument('--port', default=6669, type=int, help='port of server')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--local_rank', type=int,
                    help='local node rank for distributed learning (useless for now)')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--distributed', default=False, type=bool, help='use distributed or not. Do NOT SET THIS')

# learner config
parser_learner = parser.add_argument_group('Learner')
parser_learner.add_argument('--learner', default='', choices=['vanilla', 'trick', 'chann_cifar', 'chann_ilsvrc']) #'chann_search' is excluded for simplicity
parser_learner.add_argument('--lr', type=float, default=1e-1, help='initial learning rate')
parser_learner.add_argument('--lr_decy_type', default='multi_step', choices=['multi_step', 'cosine'], help='lr decy type for model params')
parser_learner.add_argument('--lr_min', type=float, default=2e-4, help='minimal learning rate for cosine decy')
parser_learner.add_argument('--momentum', type=float, default=0.9, help='momentum value')
parser_learner.add_argument('--nesterov', action='store_true')
parser_learner.add_argument('--weight_decay', type=float, default=2e-4, help='weight decay for resnet')
parser_learner.add_argument('--dropout_rate', default=0.0, type=float, help='dropout rate for mobilenet_v2, default: 0')

parser.add_argument('--gamma', type=float, default=0.99, help='time decay for baseline update')
parser_learner.add_argument('--flops', action='store_true', help='add flops regularization')
parser_learner.add_argument('--orthg_weight', type=float, default=0.001, help='orthognal regularization weight')
parser_learner.add_argument('--ft_schedual' , default='', choices=['follow_meta','fixed'])
parser_learner.add_argument('--ft_proj_lr' , type=float, default=0.001, help='lr in fixed ft_schedual')
parser_learner.add_argument('--norm_constraint' , default='', choices=['regularization','constraint'])
parser_learner.add_argument('--norm_weight' , type=float, default=0.01, help='norm 1 regularization weight')
parser_learner.add_argument('--updt_proj', type=int, default=10, help='intervals to update orthogonal loss')
parser_learner.add_argument('--ortho_type', default='l1', choices=['l1', 'l2'], help='type of norm for orthogonal regularization')

parser_arc = parser.add_argument_group('Architecture')
parser_arc.add_argument('--cfg', default='', help='decoding cfgs obtained by one-shot model')

# dataset params
parser_dataset = parser.add_argument_group('Dataset')
parser_dataset.add_argument('--dataset', choices=['cifar10', 'cifar100', 'ilsvrc_12'], default='cifar10', help='dataset to use')
parser_dataset.add_argument('--batch_size', type=int, default=128, help='batch_size for dataset')
parser_dataset.add_argument('--epochs', type=int, default=200, help='resnet:200')
parser_dataset.add_argument('--warmup_epochs', type=int, default=-1, help='epochs training weights only')
parser_dataset.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser_dataset.add_argument('--total_portion', type=float, default=1.0, help='portion of data will be used in searching')

# trick params
parser_trick = parser.add_argument_group("Tricks")
parser_trick.add_argument('--mixup', action='store_true')
parser_trick.add_argument('--mixup_alpha', type=float, default=1.0)
parser_trick.add_argument('--label_smooth', action='store_true')
parser_trick.add_argument('--label_smooth_eps', type=float, default=0.1)
parser_trick.add_argument('--se', action='store_true')
parser_trick.add_argument('--se_reduction', type=int, default=-1)

def set_loader(args):
  sampler = None

  if args.dataset == 'cifar10':
    if args.learner in ['vanilla']:
      loaders = cifar10_loader_train(args, num_workers=4)
  else:
    raise ValueError("Unknown dataset")

  return loaders, sampler


def set_learner(args):
  if args.learner == 'vanilla':
    return BasicLearner
  else:
    raise ValueError("Unknown learner")


def main():
  ##################### Preliminaries ##################
  args = parser.parse_args()
  os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  ngpus_per_node = torch.cuda.device_count()
  args.distributed = True if ngpus_per_node > 1 else False
  #####################################################

  try:
    if args.distributed:
      rank, world_size = init_dist(backend=args.dist_backend, master_ip=args.dist_url, port=args.port)
      args.rank = rank
      args.world_size = world_size
      print("Distributed Enabled. Rank %d initalized" % args.rank)
    else:
      print("Single model training...")

    if args.fix_random:
      # init seed within each thread
      manualSeed = args.seed
      np.random.seed(manualSeed)
      random.seed(manualSeed)
      torch.manual_seed(manualSeed)
      torch.cuda.manual_seed(manualSeed)
      torch.cuda.manual_seed_all(manualSeed)
      # NOTE: literally you should uncomment the following, but slower
      cudnn.deterministic = True
      cudnn.benchmark = False
      print('Warning: You have chosen to seed training. '
                    'This will turn on the CUDNN deterministic setting, '
                    'which can slow down your training considerably! '
                    'You may see unexpected behavior when restarting '
                    'from checkpoints.')
    else:
      cudnn.benchmark = True

    if args.rank == 0:
      # either single GPU or multi GPU with rank = 0
      print("+++++++++++++++++++++++++")
      print("torch version:", torch.__version__)
      print("+++++++++++++++++++++++++")

      # setup logging
      if args.exec_mode in ['train', 'misc']:
        args.job_id = generate_job_id()
        init_logging(os.path.join(args.save_path, '_'.join([args.model_type, args.learner]), args.job_id, 'record.log'))
      elif args.exec_mode == 'finetune':
        args.job_id = generate_job_id()
        init_logging(os.path.join(os.path.dirname(args.load_path), 'ft_record.log'))
      else:
        init_logging(os.path.join(os.path.dirname(args.load_path), args.load_path_log))

      print_args(vars(args))
      logging.info("Using GPU: "+args.gpu_id)

    # create model
    model_helper = ModelHelper()
    model = model_helper.get_model(args)
    model.cuda()

    if args.distributed:
      # share the same initialization
      broadcast_params(model)

    loaders, sampler = set_loader(args)
    learner_fn = set_learner(args)
    learner = learner_fn(model, loaders, args, device)

    if args.exec_mode == 'train':
      learner.train(sampler)
    elif args.exec_mode == 'eval':
      learner.evaluate()
    elif args.exec_mode == 'finetune':
      learner.finetune(sampler)
    elif args.exec_mode == 'misc':
      learner.misc()

    if args.distributed:
      dist.destroy_process_group()
    return 0

  except:
    traceback.print_exc()
    if args.distributed:
      dist.destroy_process_group()
    return 1

if __name__ == '__main__':
  main()
