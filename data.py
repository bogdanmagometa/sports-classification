from typing import Callable, Optional, Tuple, Any, List, Dict
import os
from contextlib import contextmanager

import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import v2
from torchvision.datasets import DatasetFolder, ImageFolder, VisionDataset
from torchvision import models
from torch.utils.data import Subset, DataLoader
import numpy as np
import pandas as pd
from PIL import Image


def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

class SportDataset(VisionDataset):
    NUM_CLASSES = 100
    META_FNAME = "release_data.csv"

    def __init__(
        self, root: str, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None,
        train: bool = True
    ):
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.meta = None
        self.root = root
        self.class_to_idx = self.find_classes(root)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.loader = pil_loader

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        fname, target = self.meta.iloc[index][['filename', 'class id']]
        path = os.path.join(self.root, 'data', 'data', fname)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.meta)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        meta_path = os.path.join(directory, SportDataset.META_FNAME)
        self.meta = pd.read_csv(meta_path)
        if self.train:
            self.meta = self.meta[self.meta['class id'] != -1]
            class_to_idx = dict(self.meta[['label', 'class id']].to_numpy())
            assert len(class_to_idx) == SportDataset.NUM_CLASSES
        else:
            self.meta = self.meta[self.meta['class id'] == -1]
            class_to_idx = {np.nan: -1}
        self.meta = self.meta[self.meta['filename'].str.endswith('jpg')]
        return class_to_idx

def display_random_samples(dataset: Subset, n = 16, fig = None, idx_to_class = None):
    """Given a SportDataset, display n random samples from it in a grid.
    """
    
    if idx_to_class is SportDataset and idx_to_class is None:
        idx_to_class = dataset.idx_to_class
    
    num_display = min(n, len(dataset))
    sample_idxes = np.random.choice(len(dataset), (num_display,), replace=False)

    if fig == None:
        fig = plt.figure()
    fig_width = int(np.ceil(np.sqrt(num_display)))
    fig_height = (num_display + fig_width - 1) // fig_width

    for i in range(num_display):
        ax = fig.add_subplot(fig_height, fig_width, i + 1)
        idx = sample_idxes[i]
        img, idx = dataset[idx]
        if idx_to_class:
            label = idx_to_class[idx]
            ax.set_title(label)
        img = img.cpu().numpy().transpose(1, 2, 0)
        ax.imshow(img)
