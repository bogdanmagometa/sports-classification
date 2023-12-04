from typing import Callable, Optional, Tuple, Any, List, Dict
import os
from contextlib import contextmanager
import random
from pprint import pprint

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
import wandb

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

def get_dataset(transform_config):
    rrc = transform_config['random_resize_crop']
    transforms = v2.Compose([
        T.ToTensor(),
        T.RandomHorizontalFlip(),
        T.RandomResizedCrop(rrc['size'], scale=rrc['scale'], ratio=rrc['ratio'], antialias=True),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = SportDataset('./hw-ucu-12023-4-100-sports-image-classification', transform=transforms, train=True)
    assert len(dataset) == 13695
    return dataset

def get_model(model_config: dict):
    # Model specific code
    if model_config['name'] == 'alexnet':
        model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        in_features = model.classifier[6].in_features
        classifier = nn.Linear(in_features, SportDataset.NUM_CLASSES)
        model.classifier[6] = classifier
        seq_blocks = [mod for mod in model.features if list(mod.parameters())]
        seq_blocks.extend([mod for mod in model.classifier if list(mod.parameters())])
    elif model_config['name'] == 'efficientnet_b2':
        model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        classifier = nn.Linear(in_features, SportDataset.NUM_CLASSES)
        model.classifier[1] = classifier
        seq_blocks = list(model.features)
        seq_blocks.append(model.classifier)        
    elif model_config['name'] == 'vit_h_14':
        model = models.vit_h_14(weights=models.ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1)
        in_features = model.heads.head.in_features
        classifier = nn.Linear(in_features, SportDataset.NUM_CLASSES)
        model.heads.head = classifier
        seq_blocks = list(model.encoder.layers)
        seq_blocks.append(model.heads)
    elif model_config['name'] == 'vit_b_16':
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1)
        in_features = model.heads.head.in_features
        classifier = nn.Linear(in_features, SportDataset.NUM_CLASSES)
        model.heads.head = classifier
        seq_blocks = list(model.encoder.layers)
        seq_blocks.append(model.heads)
    else:
        raise ValueError(f"Invalid model config name: {model_config['name']}")
    model.train()

    # General code
    strat = model_config['transfer_strategy']
    if strat['type'] == 'finetune':
        pass
    elif strat['type'] == 'extractor':
        for param in model.parameters():
            if param is not classifier.weight and param is not classifier.bias:
                param.requires_grad = False
    elif strat['type'] == 'random':
        freeze_prob = strat['freeze_prob']
        with temp_seed(strat['seed']):
            for param in model.parameters():
                if np.random.rand() < freeze_prob:
                    param.requires_grad = False
    elif strat['type'] == 'unfreeze_last_percent':
        num_unfreeze = round(strat['percentage'] * len(seq_blocks))
        for blk in seq_blocks[:max(len(seq_blocks) - num_unfreeze, 0)]:
            for param in blk.parameters():
                param.requires_grad = False
    else:
        raise ValueError(f"Invalid transfer strategy type: {strat['type']}")

    return model

def get_optimizer(optimizer_config: dict):
    kwargs = optimizer_config['kwargs']
    if optimizer_config['name'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), **kwargs)
    elif optimizer_config['name'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), **kwargs)
    elif optimizer_config['name'] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), **kwargs)
    else:
        raise ValueError(f"Invalid optimizer config name: {optimizer_config['name']}")

    return optimizer

NUM_EPOCHS = 100

def random_transfer_strategy():
    strategy = {}
    
    strategies_types = ['finetune', 'extractor', 'random', 'unfreeze_last_percent']
    strategy['type'] = 'unfreeze_last_percent'
    # strategy['type'] = np.random.choice(strategies_types)
    if strategy['type'] == 'unfreeze_last_percent':
        percentage = np.random.rand() * 0.1
        strategy['percentage'] = percentage
    elif strategy['type'] == 'random':
        strategy['freeze_prob'] = np.random.rand()
        strategy['seed'] = np.random.randint(0, 10000)

    return strategy

def random_optimizer():
    optimizer_names = ['Adam', 'SGD', 'AdamW']
    optimizer = {
        'name': np.random.choice(optimizer_names),
        'kwargs': {}
    }
    # if optimizer['name'] == 'SGD':
    optimizer['kwargs']['lr'] = np.random.choice([0.1, 0.01, 0.001, 0.0001, 1e-5])
    optimizer['kwargs']['weight_decay'] = np.random.choice([1e-2, 1e-3, 1e-4, 0.0])
    return optimizer

def generate_random_config(model_name: Optional[str] = None):
    # model
    model_names = ['alexnet', 'efficientnet_b2', 'vit_h_14']
    if model_name is None:
        model_name = np.random.choice(model_names)
    # strategy
    config = {
        # 'batch_size': np.random.choice([128, 256, 512]),
        'batch_size': np.random.choice([256, 512, 1024]),
        # 'batch_size': np.random.choice([1024, 2048, 4096]),
        'val_batch_size': 128,
        'randomness': {
            'seed': 0,
            'model_seed': 42,
            'split_seed': 42
        },
        "transform": {
            'random_resize_crop': {
                'size': (224, 224),
                'scale': (0.7, 1.0),
                'ratio': (0.8, 1.2)
            }
        },
        'optimizer': random_optimizer(),
        'model': {
            'name': model_name,
            'transfer_strategy': random_transfer_strategy(),
        }
    }
    return config

if __name__ == "__main__":
    # config = {
    #     'batch_size': 256, # Use only divisible by 16
    #     'val_batch_size': 128,
    #     'randomness': {
    #         'seed': 0,
    #         'model_seed': 42,
    #         'split_seed': 42
    #     },
    #     'optimizer': {
    #         'name': 'Adam',
    #         'kwargs': {}
    #     },
    #     'model': {
    #         'name': 'vit_h_14',
    #         'transfer_strategy': {
    #             'type': 'unfreeze_last_percent',
    #             'percentage': .1
    #         }
    #     }
    # }
    config = {
        "model": {
            "name": "vit_b_16",
            "transfer_strategy": {
                "type": "finetune"
            }
        },
        "optimizer": {
            "name": "Adam",
            "kwargs": {
                "lr": 0.001,
                "weight_decay": 0.000
            }
        },
        "batch_size": 512,
        "randomness": {
            "seed": 0,
            "model_seed": 42,
            "split_seed": 42
        },
        "transform": {
            'random_resize_crop': {
                'size': (224, 224),
                'scale': (0.7, 1.0),
                'ratio': (0.8, 1.2)
            }
        },
        "val_batch_size": 128
    }
    config = generate_random_config('vit_b_16')
    pprint(config)
    
    # Ensure deterministic behavior
    seed = config['randomness']['seed']
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    
    # Train dataset
    dataset = get_dataset(config['transform'])

    # Split into train and val parts
    with temp_seed(config['randomness']['split_seed']):
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
        torch.save(train_dataset.indices, 'train_dataset.pt')
        torch.save(val_dataset.indices, 'val_dataset.pt')
    torch.save(train_dataset[0][0], 'img.pt')
    exit()

    # Loaders
    if config['model']['name'] == 'alexnet':
        batch_size = config['batch_size']
    elif config['model']['name'] == 'efficientnet_b2':
        batch_size = 16
    elif config['model']['name'] == 'vit_h_14':
        batch_size = 16
    elif config['model']['name'] == 'vit_b_16':
        batch_size = 64
    acc_grad_batches = config['batch_size'] // batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=5, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['val_batch_size'], num_workers=5, shuffle=False, drop_last=False, pin_memory=True)

    # Model
    with temp_seed(config['randomness']['model_seed']):
        model = get_model(config['model'])
    model.cuda();

    optimizer = get_optimizer(config['optimizer'])

    wandb.init(
        entity="bohdanmahometa",
        project="sports-classification-4",
        resume='never',
        config=config,
        # name='noble-lake-rrc2',
        job_type='vitb16'
    )
    config = wandb.run.config
    wandb.run.log_code('.')

    # val_loss, val_acc = val_step(model, val_loader)
    # print(f"Preliminary val:\t{val_loss:.2f} loss \t {val_acc:.3f} acc\n")
    # wandb.log({
    #     'val_loss': val_loss,
    #     'val_acc': val_acc,
    #     'epoch': 0
    # })

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, threshold=0.01)

    best_val_acc = float('-inf')

    for epoch in range(1, NUM_EPOCHS + 1):
        lr = optimizer.param_groups[0]['lr']
        if lr < 1e-6:
            break

        # Log LR
        print(f"lr = {lr}")
        wandb.log({
            'lr': lr
        }, commit=False)

        model.train()
        train_loss = torch.tensor(0.0, dtype=torch.float32, device='cuda')
        train_count = 0
        train_correct = torch.tensor(0, device='cuda')
        for it, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.cuda(), labels.cuda()
            logits = model(imgs)
            loss = F.cross_entropy(logits, labels)
            if it % acc_grad_batches == 0:
                optimizer.zero_grad()
            loss.backward()
            if (it + 1) % acc_grad_batches == 0:
                optimizer.step()

            train_count += len(imgs)
            with torch.no_grad():
                train_loss += loss * len(imgs)
                train_correct += torch.sum(torch.argmax(logits, axis=1) == labels)

        # Log train metrics
        train_loss = train_loss.item() / train_count
        train_acc = train_correct.item() / train_count
        print(f"Train epoch {epoch}:\t{train_loss:.2f} loss \t {train_acc:.3f} acc")
        wandb.log({
            'train_loss': train_loss,
            'train_acc': train_acc,
            'epoch': epoch
        }, commit=False)

        # Log val metrics
        val_loss, val_acc = val_step(model, val_loader)
        print(f"Val epoch {epoch}:\t{val_loss:.2f} loss \t {val_acc:.3f} acc\n")
        wandb.log({
            'val_loss': val_loss,
            'val_acc': val_acc
        })

        scheduler.step(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            output_dir = os.path.join('experiments', wandb.run.name)
            os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(output_dir, f'ckpt.ckpt'))

    wandb.finish()
