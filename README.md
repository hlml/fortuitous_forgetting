# Fortuitous Forgetting in Connectionist Networks

## Introduction
This repository includes reference code for the paper Fortuitous Forgetting in Connectionist Networks (ICLR 2022). 


```
@inproceedings{
  zhou2022fortuitous,
  title={Fortuitous Forgetting in Connectionist Networks},
  author={Hattie Zhou and Ankit Vani and Hugo Larochelle and Aaron Courville},
  booktitle={International Conference on Learning Representations},
  year={2022},
  url={https://openreview.net/forum?id=ei3SY1_zYsE}
}
```

## Targeted Forgetting
This code implements the experiments on partial weight perturbations and their effects on easy or hard examples. Scripts are stored in `/targeted_forgetting`.

To run KE-style forgetting:

`python mixed_group_training.py --seed 1 --train_perc 0.1 --random_perc 0.1 --keep_perc 0.5 --train_iters 50000 --fname new_rand_reinit_train0.1_mislabel0.1 --no_wandb`

To run IMP-style forgetting:

`python mixed_group_training.py --seed 1 --train_perc 1 --random_perc 0.0 --keep_perc 0.3 --train_iters 50000 --weight_mask --reset_to_zero --rewind_to_init --margin_groups --fname new_weight_rewind_zero_train1_margin0.1 --no_wandb`


## Later Layer Forgetting
This code builds upon the [repository](https://github.com/ahmdtaha/knowledge_evolution) for Knowledge Evolution in Neural Networks. Scripts are stored in `/llf_ke`.

To run 10 generations of LLF on the Flower102 dataset:

`python train_KE_cls.py --epochs 200 --num_generations 11 --name resetlayer4_flower_resnet18 --weight_decay 0.0001 --arch Split_ResNet18 --reset_layer_name layer4 --set Flower102 --data $DATA_DIR --no_wandb`

To run 10 generations of KE:

`python train_KE_cls.py --epochs 200 --num_generations 11 --name ke_kels_flower_resnet18 --weight_decay 0.0001 --arch Split_ResNet18 --split_rate 0.8 --split_mode kels --set Flower102 --data $DATA_DIR --no_wandb`

To run 10 generations-equivalent of the long baseline on the Flower102 dataset:

`python train_KE_cls.py --epochs 2200 --num_generations 1 --name resetlayer4_flower_resnet18_long2200 --weight_decay 0.0001 --arch Split_ResNet18 --reset_layer_name layer4 --set Flower102 --eval_intermediate_tst 200 --data $DATA_DIR --no_wandb`

To run freeze later layers experiment:

`python train_KE_cls.py --epochs 200 --num_generations 11 --name resetlayer4_flower_resnet18_freeze_reset_layers --weight_decay 0.0001 --arch Split_ResNet18 --reset_layer_name layer4 --data $DATA_DIR --set Flower102 --reverse_freeze --freeze_non_reset --optimizer sgd_TEMP --no_wandb`

To run freeze early layers experiment:

`python train_KE_cls.py --epochs 200 --num_generations 11 --name resetlayer4_flower_resnet18_freeze_nonreset_layers --weight_decay 0.0001 --arch Split_ResNet18 --reset_layer_name layer4 --data $DATA_DIR --set Flower102 --freeze_non_reset --optimizer sgd_TEMP --no_wandb`

To run freeze later layers with fixed seed experiment:

`python train_KE_cls.py --epochs 200 --num_generations 11 --name resetlayer4_flower_resnet18_freeze_reset_layers --weight_decay 0.0001 --arch Split_ResNet18 --reset_layer_name layer4 --data $DATA_DIR --set Flower102 --reverse_freeze --freeze_non_reset --optimizer sgd_TEMP --seed 0 --fix_seed --no_wandb`

## Ease-of-teaching
This code builds upon the [repository](https://github.com/FushanLi/Ease-of-teaching-and-language-structure) for Ease-of-Teaching and Language Structure from Emergent Communication. Scripts are stored in `/ease_of_teaching`.

To run the no reset baseline:

`python forget_train.py --fname baseline_no_reset --seed 0 --no_wandb`

To run the reset receiver baseline:

`python forget_train.py --resetNum 50 --fname baseline_reset_receiver --seed 0 --reset_receiver --no_wandb`

To run partial balanced forgetting (PBF):

`python forget_train.py --resetNum 100 --fname same_weight_reinit_sender10_receiver10_reset100 --seed 0 --forget_sender --sender_keep_perc 0.1 --forget_receiver --receiver_keep_perc 0.1 --weight_mask --same_mask --no_wandb`

To run targeted forgettine experiments:

`python mixed_language_forget_samebatch.py --group_vars same_mask weight_mask reset_to_zero keep_perc seed trainIters train_with_reset reset_every --seed 0 --keep_perc 0.5 --fname new_rand_reinit`

`python mixed_language_forget_samebatch.py --group_vars same_mask weight_mask reset_to_zero keep_perc seed trainIters train_with_reset reset_every --seed 0 --keep_perc 0.5 --fname same_weight_zero --same_mask --weight_mask --reset_to_zero`
