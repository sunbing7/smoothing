from torchvision import transforms, datasets
from typing import *
import torch
import os
from torch.utils.data import Dataset
from data_loader import *

# set this environment variable to the location of your imagenet directory if you want to read ImageNet data.
# make sure your val directory is preprocessed to look like the train directory, e.g. by running this script
# https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
IMAGENET_LOC_ENV = "IMAGENET_DIR"

# list of all datasets
DATASETS = ["imagenet", "cifar10"]


def get_dataset(dataset, split, data_file='../data/cifar/cifar.h5', t_attack='green', target_class=0):
    """Return the dataset as a PyTorch Dataset object"""
    if dataset == "imagenet":
        return _imagenet(split)
    elif dataset == "cifar10":
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        data = CustomCifarAttackDataSet(data_file, is_train=0, t_attack=t_attack, mode='clean',
                                        target_class=0, transform=transform_test, portion='all')
        data_adv = CustomCifarAttackDataSet(data_file, is_train=0, t_attack=t_attack, mode='clean',
                                        target_class=target_class, transform=transform_test, portion='all')
        return data, data_adv
    elif dataset == "gtsrb":
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        data = CustomGTSRBAttackDataSet(data_file, is_train=0, t_attack=t_attack, mode='clean',
                                        target_class=0, transform=transform_test, portion='all')
        data_adv = CustomGTSRBAttackDataSet(data_file, is_train=0, t_attack=t_attack, mode='clean',
                                        target_class=target_class, transform=transform_test, portion='all')
        return data, data_adv
    elif dataset == "fmnist":
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        data = CustomFMNISTAttackDataSet(data_file, is_train=0, t_attack=t_attack, mode='clean',
                                        target_class=0, transform=transform_test, portion='all')
        data_adv = CustomFMNISTAttackDataSet(data_file, is_train=0, t_attack=t_attack, mode='clean',
                                        target_class=target_class, transform=transform_test, portion='all')
        return data, data_adv
    elif dataset == "mnistm":
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        data = CustomMNISTMAttackDataSet(data_file, is_train=0, t_attack=t_attack, mode='clean',
                                        target_class=0, transform=transform_test, portion='all')
        data_adv = CustomMNISTMAttackDataSet(data_file, is_train=0, t_attack=t_attack, mode='clean',
                                        target_class=target_class, transform=transform_test, portion='all')
        return data, data_adv
    elif dataset == "asl":
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        data = datasets.ImageFolder(root=data_file + '/test', transform=transform_test)
        data_adv = CustomASLAttackDataSet(data_file, is_train=0, t_attack=t_attack, mode='adv',
                                          target_class=target_class, transform=transform_test, portion='all')
        return data, data_adv
    elif dataset == "caltech":
        transform_test = transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        data = datasets.ImageFolder(root=data_file + '/test', transform=transform_test)
        data_adv = CustomCALTECHAttackDataSet(data_file, is_train=0, t_attack=t_attack, mode='adv',
                                          target_class=target_class, transform=transform_test, portion='all')
        return data, data_adv
    else:
        print('Unsupported dataset!')

def get_num_classes(dataset: str):
    """Return the number of classes in the dataset. """
    if dataset == "imagenet":
        return 1000
    elif dataset == "cifar10":
        return 10
    elif dataset == "fmnist":
        return 10
    elif dataset == "mnistm":
        return 10
    elif dataset == "gtsrb":
        return 43
    elif dataset == "asl":
        return 29
    elif dataset == "caltech":
        return 101



def get_normalize_layer(dataset: str) -> torch.nn.Module:
    """Return the dataset's normalization layer"""
    if dataset == "imagenet":
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
    elif dataset == "cifar10":
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV)


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]

_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]


def _cifar10(split: str) -> Dataset:
    if split == "train":
        return datasets.CIFAR10("./dataset_cache", train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]))
    elif split == "test":
        return datasets.CIFAR10("./dataset_cache", train=False, download=True, transform=transforms.ToTensor())


def _imagenet(split: str) -> Dataset:
    if not IMAGENET_LOC_ENV in os.environ:
        raise RuntimeError("environment variable for ImageNet directory not set")

    dir = os.environ[IMAGENET_LOC_ENV]
    if split == "train":
        subdir = os.path.join(dir, "train")
        transform = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    elif split == "test":
        subdir = os.path.join(dir, "val")
        transform = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
    return datasets.ImageFolder(subdir, transform)


class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: List[float], sds: List[float]):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means).cuda()
        self.sds = torch.tensor(sds).cuda()

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means) / sds
