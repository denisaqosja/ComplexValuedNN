import os
import torch
import logging
import os
import functools
import logging
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import modules

logger = logging.getLogger('base')


####################
# initialize
####################


def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    logger.info('Initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(
            weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            'initialization method [{:s}] not implemented'.format(init_type))
    

def save_network(opt, netG, optG, epoch, iter_step, checkpoint_name):
    #savepath = os.path.join(opt['path']['checkpoint'], 'I{}_E{}.pth'.format(iter_step, epoch))
    savepath = os.path.join(opt['path']['checkpoint'], checkpoint_name)

    torch.save({
        "epoch": epoch,
        "iter": iter_step,
        "model_state_dict": netG.state_dict(),
        "optimizer_state_dict": optG.state_dict()},
        savepath
    )

    return


def load_network(netG, opt, optG):
    """ Load network by resuming the training """
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")

    checkpoint_path = os.path.join(opt['path']['experiment_root'], "checkpoint", f"{opt['checkpoint']}")
    logger.info('Loading pretrained model for G [{:s}] ...'.format(checkpoint_path))

    checkpoint = torch.load(checkpoint_path)
    netG.load_state_dict(checkpoint["model_state_dict"], map_location=device, strict=not opt['model']['finetune_norm'])

    optG.load(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    iter_step = checkpoint["iter"]

    return netG, optG, epoch, iter_step


def log_architecture(model, exp_path, fname="model_architecture.txt"):
    """
    Printing architecture modules into a txt file
    """
    assert fname[-4:] == ".txt", "ERROR! 'fname' must be a .txt file"
    savepath = os.path.join(exp_path, fname)

    # getting all_params
    with open(savepath, "w") as f:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        f.write(f"Total Params: {num_params}")

    for i, layer in enumerate(model.children()):
        if isinstance(layer, torch.nn.Module):
            log_module(module=layer, exp_path=exp_path, fname=fname)
    return


def log_module(module, exp_path, fname="model_architecture.txt", append=True):
    """
    Printing architecture modules into a txt file
    """
    assert fname[-4:] == ".txt", "ERROR! 'fname' must be a .txt file"
    savepath = os.path.join(exp_path, fname)

    # writing from scratch or appending to existing file
    if append is False:
        with open(savepath, "w") as f:
            f.write("")
    else:
        with open(savepath, "a") as f:
            f.write("\n\n")

    # writing info
    with open(savepath, "a") as f:
        num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        f.write(f"Params: {num_params}")
        f.write("\n")
        f.write(str(module))

    return
#
