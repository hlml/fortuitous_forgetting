import numpy as np

__all__ = ["multistep_lr", "cosine_lr", "constant_lr", "get_policy",'step_lr','long_cosine_lr']


def get_policy(name):
    if name is None:
        return constant_lr

    out_dict = {
        "constant_lr": constant_lr,
        "val_dependent_lr": val_dependent_lr,
        "cosine_lr": cosine_lr,
        "multistep_lr": multistep_lr,
        "step_lr": step_lr,
        "long_cosine_lr":long_cosine_lr,
    }

    return out_dict[name]


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def constant_lr(optimizer, args, **kwargs):
    def _lr_adjuster(epoch, iteration):
        if epoch < args.warmup_length:
            lr = _warmup_lr(args.lr, args.warmup_length, epoch)
        else:
            lr = args.lr

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster


def val_dependent_lr(optimizer, args, **kwargs):
    def _lr_adjuster(lr, factor):
        if factor is not None:
            lr = lr / factor
            assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster


def cosine_lr(optimizer, args, **kwargs):
    def _lr_adjuster(epoch, iteration):
        if epoch < args.warmup_length:
            lr = _warmup_lr(args.lr, args.warmup_length, epoch)
        else:
            e = epoch - args.warmup_length
            es = args.epochs - args.warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * args.lr

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster

def long_cosine_lr(optimizer, current_gen, args, **kwargs):
    def _lr_adjuster(epoch, iteration):
        if epoch < args.warmup_length:
            lr = _warmup_lr(args.lr, args.warmup_length, epoch)
        else:
            e = epoch + (args.epochs * current_gen) - args.warmup_length
            es = (args.epochs * args.num_generations) - args.warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * args.lr

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster

def step_lr(optimizer, cfg, **kwargs):
    def _lr_adjuster(epoch, iteration):
        lr = cfg.lr
        #hard coded for tiny Imagenet learning schedule
        if cfg.num_generations > 1:
            if epoch < cfg.warmup_length:
                lr = _warmup_lr(cfg.lr, cfg.warmup_length, epoch)

            if epoch >= 80 < 120:
                lr /= 10
            elif epoch >= 120:
                lr /= 100
        else:
            if epoch < cfg.warmup_length:
                lr = _warmup_lr(cfg.lr, cfg.warmup_length, epoch)
            if (80 * (cfg.epochs / 160)) <= epoch < (120 * (cfg.epochs / 160)):
                lr /= 10
            elif epoch >= (120 * (cfg.epochs / 160)):
                lr /= 100

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster

def multistep_lr(optimizer, args, **kwargs):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    def _lr_adjuster(epoch, iteration):
        lr = args.lr * (args.lr_gamma ** (epoch // args.lr_adjust))

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster


def _warmup_lr(base_lr, warmup_length, epoch):
    return base_lr * (epoch + 1) / warmup_length
