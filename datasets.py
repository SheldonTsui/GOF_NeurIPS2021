"""Datasets"""

import os

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import torchvision.transforms as transforms
import glob
import PIL


class CelebA(Dataset):
    """CelebA Dataset"""

    def __init__(self, img_size, **kwargs):
        super().__init__()

        dataset_path = './data/celeba/img_align_celeba/*.jpg'
        self.data = glob.glob(dataset_path)
        assert len(self.data) > 0, "Can't find data; make sure you specify the path to your dataset"
        self.transform = transforms.Compose([
            transforms.Resize(320),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((img_size, img_size), interpolation=0)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = PIL.Image.open(self.data[index])
        X = self.transform(X)

        return X, 0

class BFM(Dataset):
    """BFM Dataset"""

    def __init__(self, img_size, **kwargs):
        super().__init__()

        dataset_path = './data/BFM/train/image/*.png'
        self.data = glob.glob(dataset_path)
        assert len(self.data) > 0, "Can't find data; make sure you specify the path to your dataset"
        self.transform = transforms.Compose([
            transforms.CenterCrop(170),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((img_size, img_size), interpolation=0)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = PIL.Image.open(self.data[index])
        X = self.transform(X)

        return X, 0

class Cats(Dataset):
    """Cats Dataset"""

    def __init__(self, img_size, **kwargs):
        super().__init__()
        
        dataset_path = './data/cats/*.jpg'
        self.data = glob.glob(dataset_path)
        assert len(self.data) > 0, "Can't find data; make sure you specify the path to your dataset"
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=0),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
            transforms.RandomHorizontalFlip(p=0.5)
        ])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        X = PIL.Image.open(self.data[index])
        X = self.transform(X)
        
        return X, 0

class Carla(Dataset):
    """Carla Dataset"""

    def __init__(self, img_size, **kwargs):
        super().__init__()
        
        dataset_path = './data/carla/*.png'
        self.data = glob.glob(dataset_path)
        assert len(self.data) > 0, "Can't find data; make sure you specify the path to your dataset"
        self.transform = transforms.Compose(
                    [transforms.Resize((img_size, img_size), interpolation=0), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        X = PIL.Image.open(self.data[index])
        X = self.transform(X)
        
        return X, 0


def get_dataset(name, subsample=None, batch_size=1, **kwargs):
    dataset = globals()[name](**kwargs)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
        num_workers=8
    )
    return dataloader, 3

def get_dataset_distributed(name, world_size, rank, batch_size, **kwargs):
    dataset = globals()[name](**kwargs)

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        num_workers=16,
    )

    return dataloader, 3
