import os
import sys
import yaml
import argparse
import os.path as osp
import logging.config
from utils import os_utils
from utils import log_utils
from utils import path_utils

args = None

class Config:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Knowledge Evolution Training Approach")

        # General Config
        parser.add_argument(
            "--data", help="path to dataset base directory", default="/home/datasets"
        )

        parser.add_argument("--optimizer", help="Which optimizer to use", default="sgd")
        parser.add_argument("--set", type=str, default="Flower102",
                            choices=['Flower102Pytorch', 'Flower102', 'CUB200', 'Aircrafts', 'Dog120', 'MIT67',
                                     'CIFAR10', 'CIFAR10val', 'CIFAR100', 'CIFAR100val', 'tinyImagenet_full', 'tinyImagenet_val',
                                     'CUB200_val', 'Dog120_val', 'MIT67_val'])

        parser.add_argument(
            "-a", "--arch", metavar="ARCH", default="Split_ResNet18", help="model architecture",
            choices=['Split_ResNet18', 'Split_ResNet18Norm', 'Split_ResNet34', 'Split_ResNet50', 'Split_ResNet50Norm',
                     'Split_ResNet101', 'Split_googlenet',
                     'Split_densenet121', 'Split_densenet161', 'Split_densenet169', 'Split_densenet201', 'vgg11', 'vgg11_bn'
                     ]
        )
        parser.add_argument(
            "--config_file", help="Config file to use (see configs dir)", default=None
        )
        parser.add_argument(
            "--log-dir", help="Where to save the runs. If None use ./runs", default=None
        )

        parser.add_argument(
            '--evolve_mode', default='rand', choices=['rand', 'zero'],
            help='How to initialize the reset-hypothesis.')

        parser.add_argument(
            "-t",
            "--num_threads",
            default=10,
            type=int,
            metavar="N",
            help="number of data loading workers (default: 10)",
        )
        parser.add_argument(
            "--epochs",
            default=200,
            type=int,
            metavar="N",
            help="number of total epochs to run",
        )

        parser.add_argument(
            "--eval_intermediate_tst",
            default=0,
            type=int,
            help="eval tst every N epochs instead of evaluating at the end of each generation",
        )        
        
        parser.add_argument(
            "--start-epoch",
            default=None,
            type=int,
            metavar="N",
            help="manual epoch number (useful on restarts)",
        )
        parser.add_argument(
            "-b",
            "--batch_size",
            default=32,
            type=int,
            metavar="N",
            help="mini-batch size",
        )
        parser.add_argument(
            "--lr",
            "--learning-rate",
            default=0.256,
            type=float,
            metavar="LR",
            help="initial learning rate",
            dest="lr",
        )
        parser.add_argument(
            "--warmup_length", default=5, type=int, help="Number of warmup iterations"
        )
        parser.add_argument(
            "--momentum", default=0.9, type=float, metavar="M", help="momentum"
        )
        parser.add_argument(
            "--wd",
            "--weight_decay",
            default=1e-4,
            type=float,
            metavar="W",
            help="weight decay (default: 1e-4)",
            dest="weight_decay",
        )
        parser.add_argument(
            "-p",
            "--print-freq",
            default=10000,
            type=int,
            metavar="N",
            help="print frequency",
        )
        parser.add_argument('--samples_per_class', default=1, type=int,
                            help='Number of samples per class inside a mini-batch.'
                            )
        parser.add_argument('--alpha', default=32, type=float,
                            help='Scaling Parameter setting'
                            )
        parser.add_argument('--warm', default=1, type=int,
                            help='Warmup training epochs'
                            )
        parser.add_argument(
            "--resume",
            default="",
            type=str,
            metavar="PATH",
            help="path to latest checkpoint (default: none)",
        )
        
        parser.add_argument(
            "--reset_layer_name",
            default=None,
            type=str,
            help="layer to start resetting (LLF)",
        )

        parser.add_argument(
            "--pretrained",
            dest="pretrained",
            default=None,
            type=str,
            help="use pre-trained model",
        )
        parser.add_argument(
            "--seed", default=None, type=int, help="need to set fix_seed = True to take effect"
        )

        parser.add_argument(
            "--gpu",
            default='0',
            type=int,
            help="Which GPUs to use?",
        )
        parser.add_argument(
            "--test_interval", default=10, type=int, help="Eval on tst/val split every ? epochs"
        )

        # Learning Rate Policy Specific
        parser.add_argument(
            "--lr_policy", default="cosine_lr", help="Policy for the learning rate."
        )
        parser.add_argument(
            "--multistep-lr-adjust", default=30, type=int, help="Interval to drop lr"
        )
        parser.add_argument("--multistep-lr-gamma", default=0.1, type=int, help="Multistep multiplier")
        parser.add_argument(
            "--name", default=None, type=str, help="Experiment name to append to filepath"
        )
        parser.add_argument(
            "--log_file", default='train_log.txt', type=str, help="Experiment name to append to filepath"
        )
        parser.add_argument(
            "--save_every", default=-1, type=int, help="Save every ___ epochs"
        )
        parser.add_argument(
            "--num_generations", default=5, type=int, help="Number of training generations"
        )
        parser.add_argument('--lr-decay-step', default=10, type=int,help='Learning decay step setting')
        parser.add_argument('--lr-decay-gamma', default=0.5, type=float,help='Learning decay gamma setting')
        parser.add_argument(
            "--split_rate",
            default=0.5,
            help="What is the split-rate for the split-network weights?",
            type=float,
        )
        parser.add_argument(
            "--bias_split_rate",
            default=0.5,
            help="What is the bias split-rate for the split-network weights?",
            type=float,
        )

        parser.add_argument(
            "--slim_factor",
            default=1.0,
            help="This variable is used to extract a slim network from a dense network. "
                 "It is initialized using the split_rate of the trained dense network.",
            type=float,
        )
        parser.add_argument(
            "--split_mode",
            default="wels",
            choices=['kels','wels'],
            help="how to split the binary mask",
        )
        parser.add_argument(
            "--conv_type", type=str, default='SplitConv', help="SplitConv | DenseConv"
        )
        parser.add_argument(
            "--linear_type", type=str, default='SplitLinear', help="SplitLinear | DenseLinear"
        )
        parser.add_argument("--mode", default="fan_in", help="Weight initialization mode")
        parser.add_argument(
            "--nonlinearity", default="relu", help="Nonlinearity used by initialization"
        )
        parser.add_argument("--bn_type", default='SplitBatchNorm', help="BatchNorm type",
                            choices=['NormalBatchNorm','NonAffineBatchNorm','SplitBatchNorm'])
        parser.add_argument(
            "--init", default="kaiming_normal", help="Weight initialization modifications"
        )
        parser.add_argument(
            "--no-bn-decay", action="store_true", default=False, help="No batchnorm decay"
        )
        parser.add_argument(
            "--scale-fan", action="store_true", default=False, help="scale fan"
        )
        parser.add_argument("--save_model", action="store_true", default=False, help="save model checkpoints")
        parser.add_argument("--cs_kd", action="store_true", default=False, help="Enable Cls_KD")
        parser.add_argument("--reset_hypothesis", action="store_true", default=False, help="Reset mask across generations")
        parser.add_argument("--reverse_reset", action="store_true", default=False, help="reset layers BEFORE reset layer name") 
        parser.add_argument("--reverse_freeze", action="store_true", default=False, help="freeze reset layers") 
        parser.add_argument("--freeze_non_reset", action="store_true", default=False, help="freeze non-reset layers") 
        parser.add_argument("--no_wandb", action="store_true", default=False, help="no wandb")
        parser.add_argument("--group_vars", type=str, nargs='+', default="", help="variables used for grouping in wandb")

        parser.add_argument(
            "--label_smoothing",
            type=float,
            help="Label smoothing to use, default 0.0",
            default=0.1,
        )

        parser.add_argument(
            "--trainer", type=str, default="default"
        )
        parser.add_argument(
            "--reset_bn", action="store_true", default=False, help="reset bn each generation"
        )
        parser.add_argument(
            "--no_reset", action="store_true", default=False, help="do not reset weight each generation"
        )
        parser.add_argument(
            "--fix_seed", action="store_true", default=False, help="set a seed to fix reinit values"
        )
        
        self.parser = parser

    def parse(self,args):
        self.cfg = self.parser.parse_args(args)

        # Allow for use from notebook without config file
        # self.read_config_file()
        # self.read_cmd_args()

        if self.cfg.set == 'Flower102' or self.cfg.set == 'Flower102Pytorch':
            self.cfg.num_cls = 102
            self.cfg.eval_tst = True
        elif self.cfg.set == 'CUB200':
            self.cfg.num_cls = 200
            self.cfg.eval_tst = False
        elif self.cfg.set == 'ImageNet':
            self.cfg.num_cls = 1000
            self.cfg.eval_tst = False

        elif self.cfg.set in ['tinyImagenet_full', 'tinyImagenet_val']:
            self.cfg.num_cls = 200
            self.cfg.eval_tst = False

        elif self.cfg.set == 'Dog120':
            self.cfg.num_cls = 120
            self.cfg.eval_tst = False
        elif self.cfg.set == 'MIT67':
            self.cfg.num_cls = 67
            self.cfg.eval_tst = False
        elif self.cfg.set == 'Aircrafts':
            self.cfg.num_cls = 100
            self.cfg.eval_tst = True
        elif self.cfg.set == 'CIFAR10' or self.cfg.set == 'CIFAR10val':
            self.cfg.num_cls = 10
            self.cfg.eval_tst = False
        elif self.cfg.set == 'CIFAR100' or self.cfg.set == 'CIFAR100val':
            self.cfg.num_cls = 100
            self.cfg.eval_tst = False
        else:
            raise NotImplementedError('Invalid dataset {}'.format(self.cfg.set))

        if self.cfg.cs_kd:
            self.cfg.samples_per_class = 2
            self.cfg.label_smoothing = 0
            
        self.cfg.bias_split_rate = self.cfg.split_rate
        
        self.cfg.group_name = self.cfg.name
        self.cfg.name = 'SPLT_CLS_{}_{}_cskd{}_smth{}_k{}_G{}_e{}_ev{}_hReset{}_sm{}_{}_seed{}/'.format(self.cfg.set, self.cfg.arch, self.cfg.cs_kd, self.cfg.label_smoothing, self.cfg.split_rate, self.cfg.num_generations, self.cfg.epochs, self.cfg.evolve_mode, self.cfg.reset_hypothesis, self.cfg.split_mode, self.cfg.name, self.cfg.seed)

        self.cfg.exp_dir = osp.join(path_utils.get_checkpoint_dir() , self.cfg.name)

        os_utils.touch_dir(self.cfg.exp_dir)
        log_file = os.path.join(self.cfg.exp_dir, self.cfg.log_file)
        logging.config.dictConfig(log_utils.get_logging_dict(log_file))
        self.cfg.logger = logging.getLogger('KE')

        return self.cfg

