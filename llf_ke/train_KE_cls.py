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

def train_dense(cfg, generation, model=None):

    if model is None:
        model = net_utils.get_model(cfg)

    if cfg.pretrained and cfg.pretrained != 'imagenet':
        net_utils.load_pretrained(cfg.pretrained,cfg.gpu, model,cfg)
        model = net_utils.move_model_to_gpu(cfg, model)
        if not cfg.no_reset:
            net_utils.split_reinitialize(cfg,model,reset_hypothesis=cfg.reset_hypothesis)
        
    if cfg.reset_layer_name is not None and generation == 0 and not cfg.no_reset:
        print('RESETTING ENTIRE LAYERS')
        total_params = 0
        reset_params = 0
        if cfg.reverse_reset:
            start_reset_flag = 1
        else:
            start_reset_flag = 0
        for name, param in model.named_parameters():
            if cfg.reset_layer_name in name:
                if cfg.reverse_reset:
                    start_reset_flag = 0
                else:
                    start_reset_flag = 1              
            if 'mask' in name:
                print(name)
                total_params += param.nelement()
                if start_reset_flag:
                    param.data = torch.zeros_like(param)
                    reset_params += param.nelement()
                else:
                    param.data = torch.ones_like(param)

        print('resetting %.3f percent of parameters' %(reset_params/total_params*100))
    
    if generation > 0 and not cfg.no_reset:
        print('Reinitializing masked weights')
        net_utils.split_reinitialize(cfg,model,reset_hypothesis=cfg.reset_hypothesis)
        
        if cfg.reset_layer_name is not None and cfg.freeze_non_reset: #does not work with reverse reset
            if cfg.reverse_freeze:
                start_reset_flag = 1
            else:
                start_reset_flag = 0
            for name, param in model.named_parameters():
                if cfg.reset_layer_name in name:
                    if cfg.reverse_freeze:
                        start_reset_flag = 0
                    else:
                        start_reset_flag = 1
                if not start_reset_flag and 'fc' not in name and 'classifier.3' not in name and 'classifier.6' not in name: #currently hardcoded, need to change for different model archs
                    param.requires_grad = False
        
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

    return model


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

    for gen in range(cfg.num_generations):
        cfg.start_epoch = 0

        model = train_dense(cfg, gen, model=model)

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
            wandb.init(project="fortuitous_forgetting",
                   group="llf_ke")
            for var in cfg.group_vars:
                wandb.config.update({var:getattr(cfg, var)})
        else:
            wandb.init(project="fortuitous_forgetting",
                       group="llf_ke")
            for var in cfg.group_vars:
                wandb.config.update({var: getattr(cfg, var)})
                
    if cfg.seed is not None and cfg.fix_seed: #FIXING SEED LEADS TO SAME REINITIALIZATION VALUES FOR EACH GENERATION
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)              

    start_KE(cfg)
