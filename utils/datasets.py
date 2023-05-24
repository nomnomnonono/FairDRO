import os
import abc
import glob
import logging
from skimage.io import imread
import numpy as np
import pandas as pd
import random
import json

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


DIR = os.path.abspath(os.path.dirname(__file__))
DATASETS_DICT = {"celeba": "CelebA", "utkface": "UTKFace"}
DATASETS = list(DATASETS_DICT.keys())


def get_dataset(dataset, which_set, y, name=None, method=None, label_dir=None, root=None):
    dataset = dataset.lower()
    try:
        Dataset = eval(DATASETS_DICT[dataset])
        dataset = Dataset(which_set, y, name=name, method=method, label_dir=label_dir, root=root)
        return dataset
    except KeyError:
        raise ValueError("Unkown dataset: {}".format(dataset))


def get_img_size(dataset):
    return get_dataset(dataset).img_size


class CelebA(Dataset):
    img_size = (3, 64, 64)

    def __init__(self, which_set, y, abel_dir=None, root=os.path.join("data", "CelebA", "img_align_celeba"), **kwargs):
        super().__init__(root, [transforms.ToTensor()], **kwargs)
        self.which_set = which_set
        self.y = y
        if which_set == 'train':
            self.imgs = glob.glob(os.path.join(root, 'train') + '/*')
            self.imgs.sort()
            self.labels = os.path.join(os.path.join(root, 'label.txt'))
            self.labels = pd.read_csv(self.labels, sep=" ").replace(-1, 0)[:162770]
        elif which_set == 'val':
            self.imgs = glob.glob(os.path.join(root, 'val') + '/*')
            self.imgs.sort()
            self.labels = os.path.join(os.path.join(root, 'label.txt'))
            self.labels = pd.read_csv(self.labels, sep=" ").replace(-1, 0)[162770:]
        elif which_set == 'test':
            self.imgs = glob.glob(os.path.join(root, 'test') + '/*')
            self.imgs.sort()
            self.labels = os.path.join(os.path.join(root, 'label.txt'))
            self.labels = pd.read_csv(self.labels, sep=" ").replace(-1, 0)[182637:]

        self.idx_y1a1 = np.where(np.array(((self.labels["Male"]==1) & (self.labels["Smiling"]==1)).tolist()) == True)[0]
        self.idx_y1a0 = np.where(np.array((self.labels["Male"]==0) & (self.labels["Smiling"]==1).tolist()) == True)[0]
        self.idx_y0a1 = np.where(np.array((self.labels["Male"]==1) & (self.labels["Smiling"]==0).tolist()) == True)[0]
        self.idx_y0a0 = np.where(np.array((self.labels["Male"]==0) & (self.labels["Smiling"]==0).tolist()) == True)[0]
        self.num_groups = len(self.labels["Male"].unique())
        self.num_targets = len(self.labels["Smiling"].unique())

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = imread(img_path)
        img = self.transforms(img)
        sens = np.array([self.labels.iloc[idx]["Male"]], dtype="float")
        label = np.array(self.labels.iloc[idx][self.y], dtype="float")
        return img, sens, label


class UTKFace(Dataset):
    img_size = (3, 64, 64)

    def __init__(self, which_set, y, label_dir=None, root=os.path.join("data", "UTKFace"), **kwargs):
        super().__init__(root, [transforms.ToTensor()], **kwargs)
        self.which_set = which_set
        self.y = y
        if which_set == 'train':
            self.imgs = glob.glob(os.path.join(root, 'train') + '/*')
            self.imgs.sort()
            self.labels = os.path.join(os.path.join(root, 'label.csv'))
            self.labels = pd.read_csv(self.labels)[:18964]
        elif which_set == 'val':
            self.imgs = glob.glob(os.path.join(root, 'val') + '/*')
            self.imgs.sort()
            self.labels = os.path.join(os.path.join(root, 'label.csv'))
            self.labels = pd.read_csv(self.labels)[18964:]
        elif which_set == 'test':
            self.imgs = glob.glob(os.path.join(root, 'test') + '/*')
            self.imgs.sort()
            self.labels = os.path.join(os.path.join(root, 'label.csv'))
            self.labels = pd.read_csv(self.labels)[21334:]
        
        self.idx_y1a1 = np.where(np.array(((self.labels["female"]==1) & (self.labels["old"]==1)).tolist()) == True)[0]
        self.idx_y1a0 = np.where(np.array((self.labels["female"]==0) & (self.labels["old"]==1).tolist()) == True)[0]
        self.idx_y0a1 = np.where(np.array((self.labels["female"]==1) & (self.labels["old"]==0).tolist()) == True)[0]
        self.idx_y0a0 = np.where(np.array((self.labels["female"]==0) & (self.labels["old"]==0).tolist()) == True)[0]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = imread(img_path)
        img = self.transforms(img)
        sens = np.array([self.labels.iloc[idx]["female"]], dtype="float")
        label = np.array(self.labels.iloc[idx][self.y], dtype="float")
        return img, sens, label
