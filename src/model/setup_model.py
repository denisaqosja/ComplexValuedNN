import os
import torch 

from model.schedulers import LRWarmUp, ExponentialLRSchedule

def setup_optimization(exp_params, model):
    """
    Initializing the optimizer object used to update the model parameters

    Args:
    -----
    exp_params: dictionary
        parameters corresponding to the different experiment
    model: nn.Module
        instanciated neural network model
    Returns:
    --------
    optimizer: Torch Optim object
        Initialized optimizer
    scheduler: Torch Optim object
        learning rate scheduler object used to decrease the lr after some epochs
    """
    lr = exp_params["training"]["lr"]

    # filtering parameters to assign different learning rates
    parameters = [
            {"params": model.parameters(), "lr": lr}
        ]
    print(f"  --> Model learning rate {lr}")

    # setting up optimizer and LR-scheduler
    optimizer = setup_optimizer(parameters, exp_params)
    scheduler = setup_scheduler(exp_params, optimizer)

    return optimizer, scheduler


def setup_optimizer(parameters, exp_params):
    """ Instanciating a new optimizer """
    lr = exp_params["training"]["lr"]
    momentum = exp_params["training"]["momentum"]
    optimizer = exp_params["training"]["optimizer"]
    nesterov = exp_params["training"]["nesterov"]

    # SGD-based optimizer
    if optimizer == "adam":
        optimizer = torch.optim.Adam(parameters, lr=lr)
    elif optimizer == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=lr)
    else:
        optimizer = torch.optim.SGD(parameters, lr=lr, momentum=momentum,
                                    nesterov=nesterov, weight_decay=0.0005)

    return optimizer


def setup_scheduler(exp_params, optimizer):
    """ Instanciating a new scheduler """
    lr = exp_params["optimizer"]["lr"]
    lr_factor = exp_params["optimizer"]["lr_factor"]
    patience = exp_params["optimizer"]["patience"]
    scheduler = exp_params["optimizer"]["scheduler"]

    if scheduler == "plateau":
        print("Setting up Plateau LR-Scheduler:")
        print(f"  --> Patience: {patience}")
        print(f"  --> Factor:   {lr_factor}")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                patience=patience,
                factor=lr_factor,
                min_lr=1e-8,
                mode="max",
                verbose=True
            )
    elif scheduler == "step":
        print("Setting up Step LR-Scheduler")
        print(f"  --> Step Size: {patience}")
        print(f"  --> Factor:    {lr_factor}")
        scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer,
                gamma=lr_factor,
                step_size=patience
            )
    elif scheduler == "exponential":
        print("Setting up Exponential LR-Scheduler")
        print(f"  --> Init LR: {lr}")
        print(f"  --> Factor:  {lr_factor}")
        scheduler = ExponentialLRSchedule(
                optimizer=optimizer,
                init_lr=lr,
                gamma=lr_factor
            )
    else:
        print("Not using any LR-Scheduler")
        scheduler = None

    return scheduler


def update_scheduler(scheduler, exp_params, control_metric=None, iter=-1, end_epoch=False):
    """
    Updating the learning rate scheduler

    Args:
    -----
    scheduler: torch.optim
        scheduler to evaluate
    exp_params: dictionary
        dictionary containing the experiment parameters
    control_metric: float/torch Tensor
        Last computed validation metric.
        Needed for plateau scheduler
    iter: float
        number of optimization step.
        Needed for cyclic, cosine and exponential schedulers
    end_epoch: boolean
        True after finishing a validation epoch or certain number of iterations.
        Triggers schedulers such as plateau or fixed-step
    """
    scheduler_type = exp_params["optimizer"]["scheduler"]
    if scheduler_type == "plateau" and end_epoch:
        scheduler.step(control_metric)
    elif scheduler_type == "step" and end_epoch:
        scheduler.step()
    elif scheduler_type == "exponential":
        scheduler.step(iter)
    return


def setup_lr_warmup(params):
    """
    Seting up the learning rate warmup handler given experiment params

    Args:
    -----
    params: dictionary
        training parameters sub-dictionary from the experiment parameters

    Returns:
    --------
    lr_warmup: Object
        object that steadily increases the learning rate during the first iterations.

    Example:
    -------
        #  Learning rate is initialized with 3e-4 * (1/1000). For the first 1000 iterations
        #  or first epoch, the learning rate is updated to 3e-4 * (iter/1000).
        # after the warmup period, learning rate is fixed at 3e-4
        optimizer = torch.optim.Adam(params=model.parameters(), lr=3e-4)
        lr_warmup = LRWarmUp(init_lr=3e-4, warmup_steps=1000, max_epochs=1)
        ...
        lr_warmup(iter=cur_iter, epoch=cur_epoch, optimizer=optimizer)  # updating lr
    """
    use_warmup = params["lr_warmup"]
    lr = params["lr"]
    if use_warmup:
        warmup_steps = params["warmup_steps"]
        warmup_epochs = params["warmup_epochs"]
        lr_warmup = LRWarmUp(init_lr=lr, warmup_steps=warmup_steps, max_epochs=warmup_epochs)
    else:
        lr_warmup = LRWarmUp(init_lr=lr, warmup_steps=-1, max_epochs=-1)
    return lr_warmup


def save_checkpoint(path, model, optimizer, scheduler, epoch):
    filename ='checkpoint_' + str(epoch) + ".pth"
    savepath = os.path.join(path, filename)
    
    if scheduler is None:
        scheduler_data = ""  
    else:
        scheduler_data = scheduler
    
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), 
        "scheduler_state_dict": scheduler_data.state_dict(),}, 
            savepath
        )
    return 


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load(checkpoint["optimizer_state_dict"])
    scheduler.load(checkpoint["scheduler_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    
    return model, optimizer, scheduler, epoch, loss


def load_checkpoint_puqu(checkpoint_path, model, only_model=False, map_cpu=False, **kwargs):
    """
    Loading a precomputed checkpoint: state_dicts for the mode, optimizer and lr_scheduler
    Args:
    -----
    checkpoint_path: string
        path to the .pth file containing the state dicts
    model: torch Module
        model for which the parameters are loaded
    only_model: boolean
        if True, only model state dictionary is loaded
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} does not exist ...")
    if checkpoint_path is None:
        return model

    # loading model to either cpu or cpu
    if map_cpu:
        checkpoint = torch.load(checkpoint_path,  map_location="cpu")
    else:
        checkpoint = torch.load(checkpoint_path)
    # loading model parameters. Try catch is used to allow different dicts
    try:
        # our pretrained models
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception:
        # pretrained model from imagenet
        state_dict = checkpoint['model']
        state_dict["head.weight"] = model.head.weight
        state_dict["head.bias"] = model.head.bias
        model.load_state_dict(state_dict)

    # returning only model for transfer learning or returning also optimizer for resuming training
    if only_model:
        return model

    optimizer, scheduler = kwargs["optimizer"], kwargs["scheduler"]
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint["epoch"] + 1

    return model, optimizer, scheduler, epoch