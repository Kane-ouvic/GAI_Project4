import os
import torch
from torchvision import datasets, transforms
from torchvision.datasets import VisionDataset
import numpy as np
import pickle

class CIFAR10Local(VisionDataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, limit=None):
        super(CIFAR10Local, self).__init__(root, transform=transform, target_transform=target_transform)
        self.train = train  # training set or test set

        if self.train:
            self.train_data = []
            self.train_labels = []
            for batch in range(1, 6):
                file = os.path.join(root, 'data_batch_%d' % batch)
                with open(file, 'rb') as f:
                    entry = pickle.load(f, encoding='latin1')
                    self.train_data.append(entry['data'])
                    self.train_labels.extend(entry['labels'])
            self.train_data = np.vstack(self.train_data).reshape(-1, 3, 32, 32)
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
        else:
            file = os.path.join(root, 'test_batch')
            with open(file, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.test_data = entry['data']
                self.test_labels = entry['labels']
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

        if limit is not None:
            if self.train:
                self.train_data = self.train_data[:limit]
                self.train_labels = self.train_labels[:limit]
            else:
                self.test_data = self.test_data[:limit]
                self.test_labels = self.test_labels[:limit]

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        img = img.astype(np.float32)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

def get_cifar10_dataloader(data_path='./data/cifar-10-batches-py', train=True, batch_size=64, limit=None):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = CIFAR10Local(root=data_path, train=train, transform=transform, limit=limit)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader
