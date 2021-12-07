import os
import json
import argparse

import torch
import torch.nn as nn
from run_manager import RunManager
from models import VGG_CIFAR, TrainRunConfig

parser = argparse.ArgumentParser()

""" basic config """
parser.add_argument('--path', type=str, default='Exp/debug')
parser.add_argument('--gpu', help='gpu available', default='0,1,2,3')
parser.add_argument('--train', action='store_true', default=True)
parser.add_argument('--manual_seed', default=13, type=int)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--validation_frequency', type=int, default=1)
parser.add_argument('--print_frequency', type=int, default=100)

""" optimizer config """
parser.add_argument('--n_epochs', type=int, default=300)
parser.add_argument('--init_lr', type=float, default=0.1)
parser.add_argument('--lr_schedule_type', type=str, default='cosine')
parser.add_argument('--warm_epoch', type=int, default=5)
parser.add_argument('--opt_type', type=str, default='sgd', choices=['sgd'])
parser.add_argument('--momentum', type=float, default=0.9)  # opt_param
parser.add_argument('--no_nesterov', action='store_true')  # opt_param
parser.add_argument('--weight_decay', type=float, default=4e-5)
parser.add_argument('--label_smoothing', type=float, default=0.1)
parser.add_argument('--no_decay_keys', type=str,
                    default='bn#bias', choices=[None, 'bn', 'bn#bias'])

""" dataset config """
parser.add_argument('--dataset', type=str, default='cifar10',
                    choices=['cifar10', 'imagenet', 'imagenet10', 'imagenet100'])
parser.add_argument('--save_path', type=str, default='/userhome/data/cifar10')
parser.add_argument('--train_batch_size', type=int, default=256)
parser.add_argument('--test_batch_size', type=int, default=250)

""" model config """
parser.add_argument('--model', type=str, default="vgg",
                    choices=['vgg', 'resnet20', 'resnet56', 'resnet110', 'resnet18', 'resnet34', 'resnet50_TAS', 'mobilenetv2_autoslim', 'resnet50', 'mobilenetv2', 'mobilenet'])
parser.add_argument('--cfg', type=str, default="None")
parser.add_argument('--base_path', type=str, default=None)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--model_init', type=str,
                    default='he_fout', choices=['he_fin', 'he_fout'])
parser.add_argument('--init_div_groups', action='store_true')

""" runtime config """
parser.add_argument('--n_worker', type=int, default=24)
parser.add_argument('--sync_bn', action='store_true',
                    help='enabling apex sync BN.')
parser.add_argument("--local_rank", default=0, type=int)


if __name__ == '__main__':
    args = parser.parse_args()

#     random.seed(args.manual_seed)
#     torch.manual_seed(args.manual_seed)
#     torch.cuda.manual_seed(args.manual_seed)
#     torch.cuda.manual_seed_all(args.manual_seed)
#     np.random.seed(args.manual_seed)
    torch.backends.cudnn.benchmark = True
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.enabled = True
    os.makedirs(args.path, exist_ok=True)

    # distributed setting
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')
    args.world_size = torch.distributed.get_world_size()

    # prepare run config
    run_config_path = '%s/run.config' % args.path
    # build run config from args
    args.lr_schedule_param = None
    args.opt_param = {
        'momentum': args.momentum,
        'nesterov': not args.no_nesterov,
    }
    run_config = TrainRunConfig(
        **args.__dict__
    )

    if args.local_rank == 0:
        print('Run config:')
        for k, v in run_config.config.items():
            print('\t%s: %s' % (k, v))

    weight_path = None
    net_origin = None
    if args.model == "vgg":
        assert args.dataset == 'cifar10', 'vgg only supports cifar10 dataset'
        net = VGG_CIFAR(
            num_classes=run_config.data_provider.n_classes, cfg=eval(args.cfg))
        if args.base_path is not None:
            weight_path = args.base_path + '/checkpoint/model_best.pth.tar'
            net_origin = nn.DataParallel(
                VGG_CIFAR(num_classes=run_config.data_provider.n_classes))

    # build run manager
    run_manager = RunManager(args.path, net, run_config)
    if args.local_rank == 0:
        run_manager.save_config(print_info=True)

    # load checkpoints
    if args.base_path is not None:
        weight_path = args.base_path+'/checkpoint/model_best.pth.tar'
    if args.resume:
        run_manager.load_model()
        if args.train and run_manager.best_acc == 0:
            loss, acc1, acc5 = run_manager.validate(
                is_test=True, return_top5=True)
            run_manager.best_acc = acc1
    elif weight_path is not None and os.path.isfile(weight_path):
        assert net_origin is not None, "original network is None"
        net_origin.load_state_dict(torch.load(weight_path)['state_dict'])
        net_origin = net_origin.module
        if args.model == 'vgg':
            run_manager.reset_model(
                VGG_CIFAR(cfg=eval(args.cfg), cutout=True), net_origin.cpu())
    else:
        print('Random initialization')

    # train
    if args.train:
        print('Start training: %d' % args.local_rank)
        if args.local_rank == 0:
            print(net)
        run_manager.train(print_top5=True)
        if args.local_rank == 0:
            run_manager.save_model()

    output_dict = {}

    # test
    print('Test on test set')
    loss, acc1, acc5 = run_manager.validate(is_test=True, return_top5=True)
    log = 'test_loss: %f\t test_acc1: %f\t test_acc5: %f' % (loss, acc1, acc5)
    run_manager.write_log(log, prefix='test')
    output_dict = {
        **output_dict,
        'test_loss': '%f' % loss, 'test_acc1': '%f' % acc1, 'test_acc5': '%f' % acc5
    }
    json.dump(output_dict, open('%s/output' % args.path, 'w'), indent=4)
