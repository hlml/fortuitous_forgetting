#implements the comparison with https://arxiv.org/abs/2109.00267

import os
import sys
import data
import torch
import getpass
import KE_model
import importlib
import os.path as osp
import torch.nn as nn
from utils import os_utils
from utils import net_utils
from utils import csv_utils
from layers import conv_type
from utils import path_utils
from utils import model_profile
from configs.base_config import Config
import pdb
import wandb
import random
import numpy as np
import h5py
import pathlib


def get_trainer(args):
    print(f"=> Using trainer from trainers.{args.trainer}")
    trainer = importlib.import_module(f"trainers.{args.trainer}")

    return trainer.train, trainer.validate

def train_dense(cfg, generation, model=None, scale_dict=None):

    if model is None:
        model = net_utils.get_model(cfg)
    dummy_model = net_utils.get_model(cfg)
        
    if scale_dict is None:
        scale_dict = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                scale_dict[name] = torch.linalg.norm(param.data)

    if cfg.pretrained and cfg.pretrained != 'imagenet':
        net_utils.load_pretrained(cfg.pretrained,cfg.gpu, model,cfg)
        model = net_utils.move_model_to_gpu(cfg, model)
        raise NotImplementedError  # not implemented for LW
        net_utils.split_reinitialize(cfg,model,reset_hypothesis=cfg.reset_hypothesis)

    if 'ResNet18' in cfg.arch:
        layers = ['layer1.0','layer1.1','layer2.0','layer2.1','layer3.0','layer3.1','layer4.0','layer4.1','fc']
    elif 'ResNet50' in cfg.arch:
        layers = ['layer1.0','layer1.1','layer1.2','layer2.0','layer2.1','layer2.2','layer2.3','layer3.0','layer3.1',
                  'layer3.2','layer3.3','layer3.4','layer3.5','layer4.0','layer4.1','layer4.2','fc']
    else:
        raise NotImplementedError
    print("Total number of layers:", len(layers))
    for lno, layer in enumerate(layers):
        if (generation * len(layers)) // cfg.num_generations == lno:
            print('RESETTING', layer, 'plus')
            if cfg.reverse_reset:
                start_reset_flag = 1
            else:
                start_reset_flag = 0
            for (name, param), param_init in zip(model.named_parameters(), dummy_model.parameters()):
                if layer in name:
                    if cfg.reverse_reset:
                        start_reset_flag = 0
                    else:
                        start_reset_flag = 1

                if not cfg.no_rescale_weights:
                    if ('weight' in name or 'bias' in name) and not start_reset_flag:
                        cur_norm = torch.linalg.norm(param.data)
                        if scale_dict[name] > 1e-6:
                            rescale_ratio = cur_norm / scale_dict[name]
                            param.data = param.data / rescale_ratio
                        else:
                            rescale_ratio = np.inf
                            param.data *= 0.0
                        print('rescaling ' + name + ' by factor %.3f' % (rescale_ratio))
                if start_reset_flag:
                    param.data.copy_(param_init.data)

            if not cfg.no_normalize_LW and lno >= 1:
                for name, param in model.named_modules():
                    if hasattr(param, 'LW_norm'):
                        if layers[lno-1] in name:
                            param.LW_norm = True
                            param.norm.reset()
                        elif cfg.normalize_split_only:
                            param.LW_norm = False

    model = net_utils.move_model_to_gpu(cfg, model)
    #save model immediately after init
    if cfg.save_model:
        run_base_dir, ckpt_base_dir, log_base_dir = path_utils.get_directories(cfg, generation)
        net_utils.save_checkpoint(
            {
                "epoch": 0,
                "arch": cfg.arch,
                "state_dict": model.state_dict(),
    #             "optimizer": optimizer.state_dict(),
            },
            is_best=False,
            filename=ckpt_base_dir / f"init_model.state",
            save=False
        )
    
    cfg.trainer = 'default_cls'
    cfg.pretrained = None
    ckpt_path = KE_model.ke_cls_train(cfg, model, generation)

    return model, scale_dict 


def eval_slim(cfg, generation):
    original_num_epos = cfg.epochs
    # cfg.epochs = 0
    softmax_criterion = nn.CrossEntropyLoss().cuda()
    epoch = 1
    writer = None
    model = net_utils.get_model(cfg)
    net_utils.load_pretrained(cfg.pretrained, cfg.gpu, model,cfg)
    # if cfg.reset_mask:
    #     net_utils.reset_mask(cfg, model)
    model = net_utils.move_model_to_gpu(cfg, model)

    save_filter_stats = (cfg.arch in ['split_alexnet','split_vgg11_bn'])
    if save_filter_stats:
        for n, m in model.named_modules():
            if hasattr(m, "weight") and m.weight is not None:
                if hasattr(m, "mask"):
                    layer_mask = m.mask
                    if m.__class__ == conv_type.SplitConv:
                        # filter_state = [''.join(map(str, ((score_mask == True).type(torch.int).squeeze().tolist())))]
                        filter_mag = ['{},{}'.format(
                            float(torch.mean(torch.abs(m.weight[layer_mask.type(torch.bool)]))),
                            float(torch.mean(torch.abs(m.weight[(1-layer_mask).type(torch.bool)]))))
                        ]
                        os_utils.txt_write(osp.join(cfg.exp_dir, n.replace('.', '_') + '_mean_magnitude.txt'), filter_mag, mode='a+')

    dummy_input_tensor = torch.zeros((1, 3, 224, 224)).cuda()
    total_ops, total_params = model_profile.profile(model, dummy_input_tensor)
    cfg.logger.info("Dense #Ops: %f GOps" % (total_ops / 1e9))
    cfg.logger.info("Dense #Parameters: %f M (Split-Mask included)" % (total_params / 1e6))

    original_split_rate = cfg.split_rate
    original_bias_split_rate = cfg.bias_split_rate

    if cfg.split_mode == 'kels':
        cfg.slim_factor = cfg.split_rate
        cfg.split_rate = 1.0
        cfg.bias_split_rate = 1.0
        split_model = net_utils.get_model(cfg)
        split_model = net_utils.move_model_to_gpu(cfg, split_model)

        total_ops, total_params = model_profile.profile(split_model, dummy_input_tensor)
        cfg.logger.info("Split #Ops: %f GOps" % (total_ops / 1e9))
        cfg.logger.info("Split #Parameters: %f M (Split-Mask included)" % (total_params / 1e6))

        net_utils.extract_slim(split_model, model)
        dataset = getattr(data, cfg.set)(cfg)
        train, validate = get_trainer(cfg)
        last_val_acc1, last_val_acc5 = validate(dataset.tst_loader, split_model, softmax_criterion, cfg, writer, epoch)
        cfg.logger.info('Split Model : {} , {}'.format(last_val_acc1, last_val_acc5))
    else:
        last_val_acc1 = 0
        last_val_acc5 = 0

    csv_utils.write_cls_result_to_csv(
        ## Validation
        curr_acc1=0,
        curr_acc5=0,
        best_acc1=0,
        best_acc5=0,

        ## Test
        last_tst_acc1=last_val_acc1,
        last_tst_acc5=last_val_acc5,
        best_tst_acc1=0,
        best_tst_acc5=0,

        ## Train
        best_train_acc1=0,
        best_train_acc5=0,

        split_rate='slim',
        bias_split_rate='slim',

        base_config=cfg.name,
        name=cfg.name,
    )

    cfg.epochs = original_num_epos

    cfg.slim_factor = 1
    cfg.split_rate = original_split_rate
    cfg.bias_split_rate = original_bias_split_rate



def clean_dir(ckpt_dir,num_epochs):
    # print(ckpt_dir)
    if '0000' in str(ckpt_dir): ## Always keep the first model -- Help reproduce results
        return
    rm_path = ckpt_dir / 'model_best.pth'
    if rm_path.exists():
        os.remove(rm_path)

    rm_path = ckpt_dir / 'epoch_{}.state'.format(num_epochs - 1)
    if rm_path.exists():
        os.remove(rm_path)

    rm_path = ckpt_dir / 'initial.state'
    if rm_path.exists():
        os.remove(rm_path)

def start_KE(cfg):
    base_dir = pathlib.Path(f"{path_utils.get_checkpoint_dir()}/{cfg.name}")
    
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    ckpt_queue = []
    model=None
    scale_dict=None
    for gen in range(cfg.num_generations):
        cfg.start_epoch = 0
        
        model, scale_dict = train_dense(cfg, gen, model=model, scale_dict=scale_dict) 

        if cfg.num_generations == 1:
            break

if __name__ == '__main__':
    cfg = Config().parse(None)

    if not cfg.no_wandb:
        if len(cfg.group_vars) > 0:
            if len(cfg.group_vars) == 1:
                group_name = cfg.group_vars[0] + str(getattr(cfg, cfg.group_vars[0]))
            else:
                group_name = cfg.group_vars[0] + str(getattr(cfg, cfg.group_vars[0]))
                for var in cfg.group_vars[1:]:
                    group_name = group_name + '_' + var + str(getattr(cfg, var))
            wandb.init(project="llf_lw",
                   group=cfg.group_name,
                   name=group_name)
            for var in cfg.group_vars:
                wandb.config.update({var:getattr(cfg, var)})
                
    if cfg.seed is not None: #FIXING SEED SEEMS TO FIX REINITIALIZATION VALUES FOR EACH GENERATION
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)   

    start_KE(cfg)
