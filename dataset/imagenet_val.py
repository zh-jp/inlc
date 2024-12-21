from PIL import Image
from torch.utils.data import Dataset, DataLoader
import scipy.io as scio
import os
from torchvision.transforms import transforms

"""
root1: the path to the ImageNet validation dataset
root2: the path to the Webvision 1.0 dataset
"""
class ImageNetVal(Dataset):
    def __init__(self, root1='~/data/ILSVRC2012',
                 root2='~/data/webvision1.0',
                 num_classes=50, transform=None):
        self.root1 = os.path.expanduser(root1)
        self.root2 = os.path.expanduser(root2)

        self.num_classes = num_classes
        self.transform = transform
        self.selected_classes = self.load_selected_classes()
        self.selected_files = self.load_selected_files()
        self.path_list = list(self.selected_files.keys())

    def __getitem__(self, index):
        img_path = self.path_list[index]
        target = self.selected_files[img_path]
        image = Image.open(os.path.join(self.root1, 'ILSVRC2012_img_val', img_path)).convert('RGB')
        image = self.transform(image)
        return image, target

    def __len__(self):
        return len(self.selected_files)

    def load_selected_files(self):
        prefix = 'ILSVRC2012_devkit_t12/data'
        meta_path = os.path.join(self.root1, prefix, 'meta.mat')
        meta_data = scio.loadmat(meta_path)['synsets']
        idx_inf = {}
        # ground-truch -> meaning mapping, e.g. 490 -> 'n01753488'
        for i in meta_data:
            idx = i[0][0].item()
            inf = i[0][1].item()
            idx_inf[idx] = inf

        ground_truth_path = os.path.join(self.root1, prefix, 'ILSVRC2012_validation_ground_truth.txt')
        with open(ground_truth_path) as f:
            lines = f.readlines()
        label_in_imagenet = []

        for line in lines:
            idx = int(line.strip())
            label_in_imagenet.append(idx_inf[idx])

        # select the classes which are also in the Webvision dataset first num_classes classes, e.g. n01440764
        # mapping their labels to [0, num_classes-1]
        filelist = os.listdir(os.path.join(self.root1, 'ILSVRC2012_img_val'))
        selected_files = {}

        selected_classes = self.selected_classes.keys()
        for i, filename in enumerate(filelist):
            label = label_in_imagenet[i]
            if label in selected_classes:
                selected_files[filename] = self.selected_classes[label]
        return selected_files

    def load_selected_classes(self):
        info_path = os.path.join(self.root2, 'info/synsets.txt')
        with open(info_path) as f:
            lines = f.readlines()
        selected_classes = {}
        for i, line in enumerate(lines[:self.num_classes]):
            key = line.strip().split()[0]
            selected_classes[key] = i
        return selected_classes


class ImageNetValDataloader:
    def __init__(self, batch_size=64, num_classes=50, num_workers=8,
                 root1='~/data/ILSVRC2012', root2='~/data/webvision1.0'):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.num_workers = num_workers
        self.root1 = root1
        self.root2 = root2
        self.transform = transforms.Compose([
            transforms.Resize(320),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def test(self):
        dataset = ImageNetVal(self.root1, self.root2, self.num_classes, self.transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return dataloader
