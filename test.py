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

from data import SportDataset

def get_test_dataset():
    transforms = v2.Compose([
        T.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_dataset = SportDataset('./hw-ucu-12023-4-100-sports-image-classification', transform=transforms, train=False)
    assert len(test_dataset) == 797
