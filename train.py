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
from utils import temp_seed

def val_step(model, loader):
    model.eval()
    with torch.no_grad():
        val_loss = torch.tensor(0.0, dtype=torch.float32, device='cuda')
        val_count = 0
        val_correct = torch.tensor(0, device='cuda')

        for imgs, labels in loader:
            imgs, labels = imgs.cuda(), labels.cuda()
            logits = model(imgs)
            loss = F.cross_entropy(logits, labels)

            val_count += len(imgs)
            with torch.no_grad():
                val_loss += loss * len(imgs)
                val_correct += torch.sum(torch.argmax(logits, axis=1) == labels)

    return val_loss.item() / val_count, val_correct.item() / val_count

def get_dataset():
    transforms = v2.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = SportDataset('./hw-ucu-12023-4-100-sports-image-classification', transform=transforms, train=True)
    assert len(dataset) == 13695
    return dataset

NUM_EPOCHS = 100

if __name__ == "__main__":
    # Train dataset
    dataset = get_dataset()

    # Split into train and val parts
    with temp_seed(42):
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])

    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, drop_last=False, pin_memory=True)

    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    in_features = model.classifier[6].weight.shape[1]
    model.classifier[6] = nn.Linear(in_features, SportDataset.NUM_CLASSES)

    model.cuda();

    NUM_EPOCHS = 100
    optimizer = torch.optim.AdamW(model.parameters())

    val_loss, val_acc = val_step(model, val_loader)
    print(f"Preliminary val:\t{val_loss:.2f} loss \t {val_acc:.3f} acc\n")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, threshold=0.01)

    for epoch in range(NUM_EPOCHS):
        lr = optimizer.param_groups[0]['lr']
        print(f"lr = {lr}")
        if lr < 1e-6:
            break

        model.train()
        train_loss = torch.tensor(0.0, dtype=torch.float32, device='cuda')
        train_count = 0
        train_correct = torch.tensor(0, device='cuda')
        for it, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.cuda(), labels.cuda()
            logits = model(imgs)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_count += len(imgs)
            with torch.no_grad():
                train_loss += loss * len(imgs)
                train_correct += torch.sum(torch.argmax(logits, axis=1) == labels)

        print(f"Train epoch {epoch}:\t{train_loss.item() / train_count:.2f} loss \t {train_correct.item() / train_count:.3f} acc")

        val_loss, val_acc = val_step(model, val_loader)
        print(f"Val epoch {epoch}:\t{val_loss:.2f} loss \t {val_acc:.3f} acc\n")

        scheduler.step(val_loss)

