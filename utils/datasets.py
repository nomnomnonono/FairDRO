import glob
import os

import numpy as np
import pandas as pd
from skimage.io import imread
from torch.utils.data import Dataset
from torchvision import transforms

DIR = os.path.abspath(os.path.dirname(__file__))
DATASETS_DICT = {"celeba": "CelebA", "utkface": "UTKFace"}
DATASETS = list(DATASETS_DICT.keys())


def get_dataset(dataset, which_set, sens, target, root=None):
    dataset = dataset.lower()
    try:
        Dataset = eval(DATASETS_DICT[dataset])
        dataset = Dataset(which_set, sens, target, root=root)
        return dataset
    except KeyError:
        raise ValueError("Unkown dataset: {}".format(dataset))


def get_img_size(dataset):
    return get_dataset(dataset).img_size


class CelebA(Dataset):
    img_size = (3, 64, 64)

    def __init__(
        self,
        which_set,
        sens,
        target,
        root=os.path.join("data", "CelebA", "img_align_celeba"),
    ):
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.which_set = which_set
        self.y = target
        self.s = sens
        if which_set == "train":
            self.imgs = glob.glob(os.path.join(root, "train") + "/*")
            self.imgs.sort()
            self.labels = os.path.join(os.path.join(root, "train-label.txt"))
            self.labels = pd.read_csv(self.labels).replace(-1, 0)[:162770]
        elif which_set == "val":
            self.imgs = glob.glob(os.path.join(root, "val") + "/*")
            self.imgs.sort()
            self.labels = os.path.join(os.path.join(root, "label.txt"))
            self.labels = pd.read_csv(self.labels, sep=" ").replace(-1, 0)[162770:]
        elif which_set == "test":
            self.imgs = glob.glob(os.path.join(root, "test") + "/*")
            self.imgs.sort()
            self.labels = os.path.join(os.path.join(root, "label.txt"))
            self.labels = pd.read_csv(self.labels, sep=" ").replace(-1, 0)[182637:]

        self.idx_y1a1 = self.labels[
            (self.labels[self.s] == 1) & (self.labels[self.y] == 1)
        ].index.to_list()
        self.idx_y1a0 = self.labels[
            (self.labels[self.s] == 0) & (self.labels[self.y] == 1)
        ].index.to_list()
        self.idx_y0a1 = self.labels[
            (self.labels[self.s] == 1) & (self.labels[self.y] == 0)
        ].index.to_list()
        self.idx_y0a0 = self.labels[
            (self.labels[self.s] == 0) & (self.labels[self.y] == 0)
        ].index.to_list()
        self.num_groups = len(self.labels[self.s].unique())
        self.num_targets = len(self.labels[self.y].unique())

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = imread(img_path)
        img = self.transforms(img)
        sens = np.array([self.labels.iloc[idx][self.s]], dtype="float")
        label = np.array(self.labels.iloc[idx][self.y], dtype="float")
        return img, sens, label


class UTKFace(Dataset):
    img_size = (3, 64, 64)

    def __init__(self, which_set, sens, target, root=os.path.join("data", "UTKFace")):
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.which_set = which_set
        self.y = target
        self.s = sens
        if which_set == "train":
            self.imgs = glob.glob(os.path.join(root, "train") + "/*")
            self.imgs.sort()
            self.labels = os.path.join(os.path.join(root, "label.csv"))
            self.labels = pd.read_csv(self.labels)[:18964]
        elif which_set == "val":
            self.imgs = glob.glob(os.path.join(root, "val") + "/*")
            self.imgs.sort()
            self.labels = os.path.join(os.path.join(root, "label.csv"))
            self.labels = pd.read_csv(self.labels)[18964:]
        elif which_set == "test":
            self.imgs = glob.glob(os.path.join(root, "test") + "/*")
            self.imgs.sort()
            self.labels = os.path.join(os.path.join(root, "label.csv"))
            self.labels = pd.read_csv(self.labels)[21334:]

        self.idx_y1a1 = np.where(
            np.array(((self.labels[self.s] == 1) & (self.labels[self.y] == 1)).tolist())
            == True
        )[0]
        self.idx_y1a0 = np.where(
            np.array((self.labels[self.s] == 0) & (self.labels[self.y] == 1).tolist())
            == True
        )[0]
        self.idx_y0a1 = np.where(
            np.array((self.labels[self.s] == 1) & (self.labels[self.y] == 0).tolist())
            == True
        )[0]
        self.idx_y0a0 = np.where(
            np.array((self.labels[self.s] == 0) & (self.labels[self.y] == 0).tolist())
            == True
        )[0]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = imread(img_path)
        img = self.transforms(img)
        sens = np.array([self.labels.iloc[idx][self.s]], dtype="float")
        label = np.array(self.labels.iloc[idx][self.y], dtype="float")
        return img, sens, label
