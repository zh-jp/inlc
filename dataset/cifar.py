import torchvision
import numpy as np
from PIL import Image
from copy import deepcopy
from torchvision import transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from torch.utils.data import DataLoader
import torch

from dataset.imbalance import get_imb_data
from dataset.noise import noisify_cifar100_asymmetric
from dataset.imagenet32 import Imagenet32
from dataset import TwoTransforms


class CIFAR10(torchvision.datasets.CIFAR10):
    num_classes = 10

    def __init__(self, root='~/data', train=True, transform=None,
                 r_ood=0.2, r_id=0.2, r_imb=0.1, asym=False):
        print(f'using CIFAR-{self.num_classes}...')
        super().__init__(root, train=train, transform=transform)
        if train is False:
            return
        if r_imb > 0.:
            self.data, self.targets = get_imb_data(self.data, self.targets,
                                                   self.num_classes, r_imb)
            print(f'Built imbalanced dataset, r_imb={r_imb}')

        if r_ood > 0.:
            ids_ood = [i for i in range(len(self.targets)) if np.random.random() < r_ood]
            imagenet32 = Imagenet32(root='~/data/imagenet32', train=True)
            img_ood = imagenet32.data[np.random.permutation(range(len(imagenet32)))[:len(ids_ood)]]
            self.ids_ood = ids_ood
            self.data[ids_ood] = img_ood
            print(f'Mixing in OOD noise, r_ood={r_ood}')

            if r_id > 0.:
                self.ground_truth = deepcopy(self.targets)
                ids_not_ood = [i for i in range(len(self.targets)) if i not in ids_ood]
                ids_id = [i for i in ids_not_ood if np.random.random() < (r_id / (1 - r_ood))]
                if asym:
                    if self.num_classes == 10:
                        transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6, 8: 8}
                        for i, t in enumerate(self.targets):
                            if i in ids_id:
                                self.targets[i] = transition[t]
                    else:
                        self.targets = noisify_cifar100_asymmetric(self.targets, r_id)
                    print(f'Mixing in ID asym noise, r_id={r_id}')
                else:
                    for i, t in enumerate(self.targets):
                        if i in ids_id:
                            self.targets[i] = int(np.random.random() * self.num_classes)
                    print(f'Mixing in ID noise, r_id={r_id}')
                self.ids_id = ids_id

    def __getitem__(self, idx):
        image, target = self.data[idx], self.targets[idx]
        image = Image.fromarray(image)
        img = self.transform(image)
        target = torch.tensor(target).long()

        if self.train:
            return img, target, idx
        else:
            return img, target


class CIFAR100(CIFAR10):
    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [["train", "16019d7e3df5f24257cddd939b257f8d"]]
    test_list = [["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"]]
    meta = {
        "filename": "meta", "key": "fine_label_names", "md5": "7973b15100ade9c7d40fb424638fde48",
    }
    num_classes = 100


class CifarDataloader:

    def __init__(self, args=None):

        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.transform_st = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                AutoAugment(AutoAugmentPolicy.CIFAR10),
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
            ]
        )
        self.args = args
        self.num_workers = 8
        self.num_classes = args.dataset
        self.train_dataset = None

    def test(self):
        args = self.args
        test_args = {'train': False, 'transform': self.transform_val}
        if self.num_classes == 10:
            dataset = CIFAR10(**test_args)
        elif self.num_classes == 100:
            dataset = CIFAR100(**test_args)
        else:
            raise ValueError('Invalid dataset')
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=self.num_workers, pin_memory=True)
        return dataloader

    def train(self, dual=False):
        args = self.args
        train_args = {'train': True, 'r_ood': args.r_ood, 'r_id': args.r_id, 'r_imb': args.r_imb,
                      'asym': args.asym}
        if dual:
            transform = TwoTransforms(self.transform_train, self.transform_st)
        else:
            transform = self.transform_train

        if self.num_classes == 10:
            self.train_dataset = CIFAR10(transform=transform, **train_args)
        elif self.num_classes == 100:
            self.train_dataset = CIFAR100(transform=transform, **train_args)
        else:
            raise ValueError('Invalid dataset')

        dataloader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True,
                                num_workers=self.num_workers, pin_memory=True)
        return dataloader
