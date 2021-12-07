from utils import get_cifar_iter
from data_providers import DataProvider


class CifarDataProvider(DataProvider):
    def __init__(self, save_path=None, train_batch_size=256, test_batch_size=500, valid_size=None,
                 n_worker=8, manual_seed = 12, local_rank=0, world_size=1, **kwargs):

        self._save_path = save_path
        self.valid = None
        self.train = get_cifar_iter('train', self.save_path, train_batch_size, n_worker, cutout=16, manual_seed=manual_seed)
        self.test = get_cifar_iter('val', self.save_path, test_batch_size, n_worker, cutout=16, manual_seed=manual_seed)
        if self.valid is None:
            self.valid = self.test

    @staticmethod
    def name():
        return 'cifar10'

    @property
    def data_shape(self):
        return 3, 32, 32  # C, H, W

    @property
    def n_classes(self):
        return 10

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = '/userhome/data/cifar10'
        return self._save_path
