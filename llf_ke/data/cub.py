import numpy as np
import torch
from data.datasets import load_dataset


class CUB200:
    def __init__(self, cfg):

        if cfg.cs_kd:
            sampler_type = 'pair'
        else:
            sampler_type = 'default'

        if cfg.use_val:            
            trainloader, valloader, tst_loader = load_dataset('CUB200_v2', 
                                                  cfg.data, 
                                                  sampler_type, batch_size=cfg.batch_size)
            self.num_classes = trainloader.dataset.num_classes

            self.train_loader = trainloader
            self.tst_loader = tst_loader
            self.val_loader = valloader
        elif cfg.use_train_val:            
            trainloader, valloader, tst_loader = load_dataset('CUB200_v4', 
                                                  cfg.data, 
                                                  sampler_type, batch_size=cfg.batch_size)
            self.num_classes = trainloader.dataset.num_classes

            self.train_loader = trainloader
            self.tst_loader = tst_loader
            self.val_loader = valloader
        else:
            trainloader, valloader = load_dataset('CUB200', 
                                                  cfg.data, 
                                                  sampler_type, batch_size=cfg.batch_size)
            self.num_classes = trainloader.dataset.num_classes

            self.train_loader = trainloader
            self.tst_loader = valloader
            self.val_loader = valloader            