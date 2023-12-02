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

from data import SportDataset, display_random_samples

@contextmanager
def temp_seed(seed: int):
    """Set seed for both numpy and torch.
    Combine numpy and torch fork_rng.
    """
    np_state = np.random.get_state()
    np.random.seed(seed)
    try:
        with torch.random.fork_rng():
            yield
    finally:
        np.random.set_state(np_state)
