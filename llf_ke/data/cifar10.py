import numpy as np
import torch
from data.datasets import load_dataset


class CIFAR10:
    def __init__(self, cfg):

        if cfg.cs_kd:
            sampler_type = 'pair'
        else:
            sampler_type = 'default'

        trainloader, valloader = load_dataset('CIFAR10', 
                                                cfg.data, 
                                                sampler_type, batch_size=cfg.batch_size)
        self.num_classes = trainloader.dataset.num_classes

        self.train_loader = trainloader
        self.tst_loader = valloader
        self.val_loader = valloader


class CIFAR10val:
    def __init__(self, cfg):

        if cfg.cs_kd:
            sampler_type = 'pair'
        else:
            sampler_type = 'default'

        trainloader, valloader = load_dataset('CIFAR10val', 
                                                cfg.data, 
                                                sampler_type, batch_size=cfg.batch_size)
        self.num_classes = trainloader.dataset.num_classes

        self.train_loader = trainloader
        self.tst_loader = valloader
        self.val_loader = valloader