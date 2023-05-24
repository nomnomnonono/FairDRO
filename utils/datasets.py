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
COLOUR_BLACK = 0
COLOUR_WHITE = 1
DATASETS_DICT = {"celeba": "CelebA",
                 "utkface": "UTKFace",
                 "fairface": "FairFace",
                }
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


def get_background(dataset):
    return get_dataset(dataset).background_color


class DisentangledDataset(Dataset, abc.ABC):
    def __init__(self, root, transforms_list=[], logger=logging.getLogger(__name__)):
        self.root = root
        self.train_data = os.path.join(root, type(self).files["train"])
        self.transforms = transforms.Compose(transforms_list)
        self.logger = logger

        if not os.path.isdir(root):
            self.logger.info("Downloading {} ...".format(str(type(self))))
            self.download()
            self.logger.info("Finished Downloading.")

    def __len__(self):
        return len(self.imgs)

    @abc.abstractmethod
    def __getitem__(self, idx):
        pass

    @abc.abstractmethod
    def download(self):
        pass

class CelebA(DisentangledDataset):
    urls = {"train": "https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip"}
    files = {"train": "img_align_celeba"}
    img_size = (3, 64, 64)
    background_color = COLOUR_WHITE

    def __init__(self, which_set, y, name=None, method=None, label_dir=None, root=os.path.join("data", "CelebA", "img_align_celeba"), **kwargs):
        super().__init__(root, [transforms.ToTensor()], **kwargs)
        self.which_set = which_set
        self.y = y
        self.name = name
        self.split = os.path.join(os.path.join(root, 'train-val.csv'))
        self.split = pd.read_csv(self.split)
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
        else:
            pass
        self.idx_y1a1 = np.where(np.array(((self.labels["Male"]==1) & (self.labels["Smiling"]==1)).tolist()) == True)[0]
        self.idx_y1a0 = np.where(np.array((self.labels["Male"]==0) & (self.labels["Smiling"]==1).tolist()) == True)[0]
        self.idx_y0a1 = np.where(np.array((self.labels["Male"]==1) & (self.labels["Smiling"]==0).tolist()) == True)[0]
        self.idx_y0a0 = np.where(np.array((self.labels["Male"]==0) & (self.labels["Smiling"]==0).tolist()) == True)[0]
        self.num_groups = len(self.labels["Male"].unique())
        self.num_targets = len(self.labels["Smiling"].unique())
        if name is not None:
            if which_set != "test" and which_set != "val":
                self.pseudo = os.path.join(f"{label_dir}/{name}.csv")
                self.pseudo = pd.read_csv(self.pseudo)[:162770]
                dir = os.path.split(self.imgs[0])[0]

                from tqdm import trange
                if method == "fg" or method == "fp":
                    print("before:", len(self.imgs), len(self.labels))
                    for i in trange(len(self.imgs)):
                        if self.pseudo.iloc[i]["logit"] < 0.95:
                            idx = self.labels.iloc[i]["file_name"]
                            self.imgs.remove(os.path.join(dir, idx))
                    self.labels["logit"] = self.pseudo["logit"]
                    if method == "fp":
                        self.labels["Male"] = self.pseudo["label"]
                    self.labels = self.labels[self.labels["logit"]>=0.95]
                    self.labels["logit"] = 1.0
                    print("after:", len(self.imgs), len(self.labels))
                elif method == "p":
                    self.labels["Male"] = self.pseudo["label"]
                    self.labels["logit"] = 1.0
                elif method == "reweight":
                    self.labels["Male"] = self.pseudo["label"]
                    self.labels["logit"] = self.pseudo["logit"]
                elif method == "min-tau-rand-before":
                    self.labels["Male"] = self.pseudo["label"]
                    self.labels["logit"] = self.pseudo["logit"]
                    sl = name.replace("-1", "")
                    f = open(os.path.join(label_dir, "SSL2", f"celeba64.{sl}-label.json"))
                    j = json.load(f)
                    df = pd.read_csv(f"{label_dir}/{name}.csv")[:162770]
                    labeled = df.iloc[j["label"]]
                    self.labels["logit"] = (self.labels["logit"] >= labeled["logit"].min()) * 1.0
                    print(sum(self.labels["Male"]==1))
                    for i in range(len(self.labels)):
                        if (self.labels.loc[i, "logit"] == 0):
                            self.labels.loc[i, "Male"] = random.randint(0, 1)
                    #self.labels["logit"] = 1.0
                    print(sum(self.labels["Male"]==1))
            else:
                self.labels["logit"] = 1.0
        else:
            self.labels["logit"] = 1.0
                

    def download(self):
        pass

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = imread(img_path)
        img = self.transforms(img)
        sens = np.array([self.labels.iloc[idx]["Male"]], dtype="float")
        sens_logit = np.array([self.labels.iloc[idx]["logit"]], dtype="float")
        label = np.array(self.labels.iloc[idx][self.y], dtype="float")
        return img, sens, sens_logit, label


class UTKFace(DisentangledDataset):
    img_size = (3, 64, 64)
    background_color = COLOUR_WHITE
    files = {"train": ""}

    def __init__(self, which_set, y, name=None, method=None, label_dir=None, root=os.path.join("data", "UTKFace"), **kwargs):
        super().__init__(root, [transforms.ToTensor()], **kwargs)
        self.which_set = which_set
        self.y = y
        self.name = name
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
        else:
            pass

        if name is not None:
            if which_set != "test" and which_set != "val":
                self.pseudo = os.path.join(os.path.join(label_dir, f"{name}.csv"))
                self.pseudo = pd.read_csv(self.pseudo)[:18964]
                dir = os.path.split(self.imgs[0])[0]
                from tqdm import trange
                if method == "fg" or method == "fp":
                    print("before:", len(self.imgs), len(self.labels))
                    for i in trange(len(self.imgs)):
                        if self.pseudo.iloc[i]["logit"] < 0.95:
                            idx = self.labels.iloc[i]["file_name"]
                            self.imgs.remove(os.path.join(dir, idx))
                    self.labels["logit"] = self.pseudo["logit"]
                    if method == "fp":
                        self.labels["female"] = self.pseudo["label"]
                    self.labels = self.labels[self.labels["logit"]>=0.95]
                    self.labels["logit"] = 1.0
                    print("after:", len(self.imgs), len(self.labels))
                if method == "p":
                    self.labels["female"] = self.pseudo["label"]
                    self.labels["logit"] = 1.0
                elif method == "reweight":
                    self.labels["Male"] = self.pseudo["label"]
                    self.labels["logit"] = self.pseudo["logit"]
                elif method == "min-tau-rand-before":
                    self.labels["female"] = self.pseudo["label"]
                    self.labels["logit"] = self.pseudo["logit"]
                    sl = name.replace("-1", "")
                    f = open(os.path.join(label_dir, "SSL2", f"utkface.{sl}-label.json"))
                    j = json.load(f)
                    df = pd.read_csv(f"{label_dir}/{name}.csv")[:18964]
                    labeled = df.iloc[j["label"]]
                    self.labels["logit"] = (self.labels["logit"] >= labeled["logit"].min()) * 1.0
                    print(sum(self.labels["female"]==1))
                    for i in range(len(self.labels)):
                        if (self.labels.loc[i, "logit"] == 0):
                            self.labels.loc[i, "female"] = random.randint(0, 1)
                    #self.labels["logit"] = 1.0
                    print(sum(self.labels["female"]==1))
            else:
                self.labels["logit"] = 1.0
        else:
            self.labels["logit"] = 1.0
        
        self.idx_y1a1 = np.where(np.array(((self.labels["female"]==1) & (self.labels["old"]==1)).tolist()) == True)[0]
        self.idx_y1a0 = np.where(np.array((self.labels["female"]==0) & (self.labels["old"]==1).tolist()) == True)[0]
        self.idx_y0a1 = np.where(np.array((self.labels["female"]==1) & (self.labels["old"]==0).tolist()) == True)[0]
        self.idx_y0a0 = np.where(np.array((self.labels["female"]==0) & (self.labels["old"]==0).tolist()) == True)[0]

    def download(self):
        pass

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = imread(img_path)
        img = self.transforms(img)
        sens = np.array([self.labels.iloc[idx]["female"]], dtype="float")
        sens_logit = np.array([self.labels.iloc[idx]["logit"]], dtype="float")
        label = np.array(self.labels.iloc[idx][self.y], dtype="float")
        return img, sens, sens_logit, label


class FairFace(DisentangledDataset):
    img_size = (3, 64, 64)
    background_color = COLOUR_WHITE
    files = {"train": ""}

    def __init__(self, which_set, y, name=None, method=None, label_dir=None, root=os.path.join("data", "FairFace"), **kwargs):
        super().__init__(root, [transforms.ToTensor()], **kwargs)
        self.which_set = which_set
        self.y = y
        self.name = name
        self.root = root
        if which_set == 'train':
            self.labels = os.path.join(os.path.join(root, 'only-train_labels.csv'))
            self.labels = pd.read_csv(self.labels)
        elif which_set == 'mlp':
            self.labels = os.path.join(os.path.join(root, 'only-val_labels.csv'))
            self.labels = pd.read_csv(self.labels)
        elif which_set == 'val':
            self.labels = os.path.join(os.path.join(root, 'val_labels.csv'))
            self.labels = pd.read_csv(self.labels)
        elif which_set == 'test':
            self.labels = os.path.join(os.path.join(root, 'test_labels.csv'))
            self.labels = pd.read_csv(self.labels)
        else:
            pass
        self.imgs = self.labels
        if name is not None:
            if which_set != "test" and which_set != "val":
                if which_set == "mlp":
                    self.pseudo = os.path.join(f"{label_dir}/val-{name}.csv")
                else:
                    self.pseudo = os.path.join(f"{label_dir}/train-{name}.csv")
                self.pseudo = pd.read_csv(self.pseudo)
                dir = os.path.split(self.imgs[0])[0]
                from tqdm import trange
                if method == "fg" or method == "fp":
                    print("before:", len(self.labels))
                    if method == "fp":
                        self.labels["gender"] = self.pseudo["label"]
                    self.labels = self.labels[self.labels["logit"]>=0.95]
                    self.labels["logit"] = 1.0
                    print("after:", len(self.labels))
                if method == "p":
                    self.labels["gender"] = self.pseudo["label"]
                    self.labels["logit"] = 1.0
                elif method == "reweight":
                    self.labels["gender"] = self.pseudo["label"]
                    self.labels["logit"] = self.pseudo["logit"]
            else:
                self.labels["logit"] = 1.0
        else:
            self.labels["logit"] = 1.0

    def download(self):
        pass

    def __getitem__(self, idx):
        img = imread(os.path.join(self.root, self.labels.iloc[idx]["file"]))
        img = self.transforms(img)
        sens = np.array([self.labels.iloc[idx]["gender"]], dtype="float")
        sens_logit = np.array([self.labels.iloc[idx]["logit"]], dtype="float")
        label = np.array(self.labels.iloc[idx]["age"], dtype="float")
        return img, sens, sens_logit, label
