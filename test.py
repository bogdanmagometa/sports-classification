from typing import Callable, Optional, Tuple, Any, List, Dict
import os
from contextlib import contextmanager
import argparse

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
import model_loaders

def get_test_dataset():
    transforms = v2.Compose([
        T.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_dataset = SportDataset('./hw-ucu-12023-4-100-sports-image-classification', transform=transforms, train=False)
    assert len(test_dataset) == 797

class TestSportsDataset(VisionDataset):
    pass

def create_alexnet():
    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    in_features = model.classifier[6].in_features
    classifier = nn.Linear(in_features, SportDataset.NUM_CLASSES)
    model.classifier[6] = classifier
    return model

def create_efficientnet_b2():
    model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    classifier = nn.Linear(in_features, SportDataset.NUM_CLASSES)
    model.classifier[1] = classifier
    return model

def create_vih14():
    model = models.vit_h_14(weights=models.ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1)
    in_features = model.heads.head.in_features
    classifier = nn.Linear(in_features, SportDataset.NUM_CLASSES)
    model.heads.head = classifier
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type='str', required=True)
    parser.add_argument('-w', '--weights', type='str', required=True)

    test_dataset = get_test_dataset()
    
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=False, pin_memory=True)
    
    model = model_loaders.load_efficient_net()
    model.load_state_dict(parser.weights)
    model.eval()

    id2pred = {}

    i = 0
    with torch.no_grad():
        for imgs, _ in test_loader:
            fnames = test_dataset.meta.iloc[i:i+len(imgs)]['filename']
            file_ids = fnames.apply(lambda x: x[:-4])
            
            logits = model(imgs)
            pred = torch.argmax(logits, dim=1)
            for fid, p in zip(file_ids, pred):
                id2pred[fid] = p

            i += len(imgs)
    
    print(id2pred)
