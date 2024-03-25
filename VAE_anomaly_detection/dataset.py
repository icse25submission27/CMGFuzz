from typing import Tuple
import torch
from torch.utils.data import Dataset, TensorDataset
from torchvision.datasets import MNIST
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torch.optim import lr_scheduler
import numpy as np
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'dist': transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
}

class NPYImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.classes = os.listdir(root_dir)  # 获取所有类别
        self.classes.sort()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}  # 将类别映射到索引

        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for filename in os.listdir(class_dir):
                if filename.endswith('.npy'):
                    path = os.path.join(class_dir, filename)
                    self.samples.append((path, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = np.load(path)
        img = img.astype(np.float32)
        img = torch.from_numpy(img).permute(2, 0, 1)
        if self.transform is not None:
            img = self.transform(img)
        
        
        return img, target
    
class ADVImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.classes = os.listdir(root_dir)  # 获取所有类别
        self.classes.sort()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}  # 将类别映射到索引

        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for filename in os.listdir(class_dir):
                if filename.endswith('.npy'):
                    path = os.path.join(class_dir, filename)
                    self.samples.append((path, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = np.load(path)
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
        if self.transform is not None:
            img = self.transform(img)
        
        
        return img, target


def rand_dataset(num_rows=60_000, num_columns=100) -> Dataset:
    return TensorDataset(torch.rand(num_rows, num_columns))


def mnist_dataset(train=True) -> Dataset:
    """
    Returns the MNIST dataset for training or testing.
    
    Args:
    train (bool): If True, returns the training dataset. Otherwise, returns the testing dataset.
    
    Returns:
    Dataset: The MNIST dataset.
    """
    return MNIST(root='./data', train=train, download=True, transform=None)

def flw17_dataset(data_dir, train=True):
    if train:
        x='train'
    else:
        x='test'
    return datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'test']}
def flw17_dataset_add(data_dir):
    x='test'
    return datasets.ImageFolder(data_dir, data_transforms[x])
def flw17_dataset_dist(data_dir,train=True):
    x='dist'
    return NPYImageDataset(data_dir)
def flw17_dataset_bim(data_dir,train=True):
    x='dist'
    return ADVImageDataset(data_dir)
def in100_dataset(data_dir, train=True):
    if train:
        x='train'
    else:
        x='test'
    return datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])

def food101_dataset(data_dir, train=True):
    if train:
        x='train'
    else:
        x='test'
    return datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])