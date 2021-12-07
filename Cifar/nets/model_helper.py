from nets.resnet_20s_decode import *
from utils.compute_flops import *
import torch


class ModelHelper(object):
  def __init__(self):
    self.model_dict = {'resnet_decode': resnet_decode}

  def get_model(self, args):
    if  args.model_type not in self.model_dict.keys():
      raise ValueError('Wrong model type.')

    num_classes = self.__get_number_class(args.dataset)

    if 'decode' in args.model_type:
      if args.cfg == '':
        raise ValueError('Running decoding model. Empty cfg!')
      cfg = [int(v) for v in args.cfg.split(',')]
      model = self.model_dict[args.model_type](cfg, num_classes, args.se, args.se_reduction)
      if args.rank == 0:
        print(model)
        print("Flops and params:")
        resol = 32 if args.dataset == 'cifar10' else 224
        print_model_param_nums(model)
        print_model_param_flops(model, resol, multiply_adds=False)

    return model

  def __get_number_class(self, dataset):
    # determine the number of classes
    if dataset == 'cifar10':
      num_classes = 10
    elif dataset == 'cifar100':
      num_classes = 100
    elif dataset == 'ilsvrc_12':
      num_classes = 1000
    return num_classes

def test():
  mh = ModelHelper()
  for k in mh.model_dict.keys():
    print(mh.get_model(k))


if __name__ == '__main__':
  test()

