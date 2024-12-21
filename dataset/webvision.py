from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from dataset import TwoTransforms
from PIL import Image
import os


class Webvision(Dataset):
    def __init__(self, root='~/data/webvision1.0', train=True, transform=None, num_classes=50):
        root = os.path.expanduser(root)
        self.root = root
        self.transform = transform
        self.train = train
        self.num_classes = num_classes
        if train:
            txt_file = 'info/train_filelist_google.txt'
        else:
            txt_file = 'info/val_filelist.txt'

        with open(os.path.join(root, txt_file)) as f:
            lines = f.readlines()
        data, targets = [], []
        for line in lines:
            img, target = line.split()
            target = int(target)
            if target < num_classes:
                data.append(img)
                targets.append(target)
        assert len(data) == len(targets)
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        img_path = self.data[index]
        target = self.targets[index]
        if self.train:
            image = Image.open(os.path.join(self.root, img_path)).convert('RGB')
        else:
            image = Image.open(os.path.join(self.root, 'val_images_256', img_path)).convert('RGB')
        image = self.transform(image)
        if self.train:
            return image, target, index
        return image, target


class WebvisionDataloader:
    def __init__(self, batch_size=64, num_classes=50, num_workers=8, root='~/data/webvision1.0'):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.num_workers = num_workers
        self.root = root

        self.transform_train = transforms.Compose([
            transforms.Resize(320),
            transforms.RandomResizedCrop(299),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.transform_st = transforms.Compose([
            transforms.Resize(320),
            transforms.RandomResizedCrop(299),
            transforms.RandomHorizontalFlip(),
            AutoAugment(AutoAugmentPolicy.IMAGENET),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.transform_test = transforms.Compose([
            transforms.Resize(320),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def train(self, dual=False):
        if dual:
            transform = TwoTransforms(self.transform_train, self.transform_st)
        else:
            transform = self.transform_train
        dataset = Webvision(root=self.root, train=True, transform=transform,
                            num_classes=self.num_classes)
        dataloader = DataLoader(
            dataset=dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers, pin_memory=True)

        return dataloader

    def test(self):
        dataset = Webvision(root=self.root, train=False, transform=self.transform_test,
                            num_classes=self.num_classes)

        dataloader = DataLoader(
            dataset=dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers, pin_memory=True)
        return dataloader
