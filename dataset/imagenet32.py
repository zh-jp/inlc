import os
import pickle
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

_train_list = ['train_data_batch_1',
               'train_data_batch_2',
               'train_data_batch_3',
               'train_data_batch_4',
               'train_data_batch_5',
               'train_data_batch_6',
               'train_data_batch_7',
               'train_data_batch_8',
               'train_data_batch_9',
               'train_data_batch_10']
_val_list = ['val_data']


def get_dataset(transform_train, transform_test):
    # prepare datasets

    # Train set
    train = Imagenet32(train=True, transform=transform_train)  # Load all 1000 classes in memory

    # Test set
    test = Imagenet32(train=False, transform=transform_test)  # Load all 1000 test classes in memory

    return train, test


class Imagenet32(Dataset):
    def __init__(self, root='~/data/imagenet32', train=True, transform=None):
        root = os.path.expanduser(root)
        self.transform = transform
        size = 32
        # Now load the picked numpy arrays

        if train:
            data, labels = [], []

            for f in _train_list:
                file = os.path.join(root, f)

                with open(file, 'rb') as fo:
                    entry = pickle.load(fo, encoding='latin1')
                    data.append(entry['data'])
                    labels += entry['labels']
            data = np.concatenate(data)

        else:
            f = _val_list[0]
            file = os.path.join(root, f)
            with open(file, 'rb') as fo:
                entry = pickle.load(fo, encoding='latin1')
                data = entry['data']
                labels = entry['labels']

        data = data.reshape((-1, 3, size, size))
        self.data = data.transpose((0, 2, 3, 1))  # Convert to HWC
        labels = np.array(labels) - 1
        self.labels = labels.tolist()

    def __getitem__(self, index):

        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index

    def __len__(self):
        return len(self.data)
