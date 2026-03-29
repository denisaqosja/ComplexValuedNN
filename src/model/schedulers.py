import math
import torch
import model.model_utils as model_utils
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, warm_epoch, start_lr=0, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = warm_epoch
        self.start_lr = start_lr
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_epoch = None
        self.base_lrs = None
        super().__init__(optimizer)

    def get_lr(self):
         # ------------------------------------------ After Warmup phase ------------------------------------------
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        # return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

        # --------------------------------------------- Warmup phase ----------------------------------------------
        # return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
        progress = float(self.last_epoch) / float(self.total_epoch)
        if self.start_lr is None:
            # default behaviour (original)
            start_lrs = [base_lr for base_lr in self.base_lrs]
        else:
            start_lrs = [self.start_lr for _ in self.base_lrs]
        target_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [start + (target - start) * progress for start, target in zip(start_lrs, target_lrs)]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
            if epoch is None:
                epoch = self.last_epoch + 1
            self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
            if self.last_epoch <= self.total_epoch:
                warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
                for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                    param_group['lr'] = lr
            else:
                if epoch is None:
                    self.after_scheduler.step(metrics, None)
                else:
                    self.after_scheduler.step(metrics, epoch - self.total_epoch)


    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self.last_epoch = self.after_scheduler.last_epoch + self.total_epoch + 1
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


class Freezer:
    def __init__(self, model, frozen_epochs=10):
        self.model = model
        self.frozen_epochs = frozen_epochs
        self.is_frozen = False

    def __call__(self, epoch):
        if epoch < self.frozen_epochs and self.is_frozen is False:
            print(f"---> Freezing the backbone {epoch } < {self.frozen_epochs} ...")
            # freeze the whole model
            model_utils.freeze_parameters(self.model)
            # Unfreeze the last classification layer
            model_utils.unfreeze_parameters(self.model.head)
            self.is_frozen = True

        elif epoch >= self.frozen_epochs and self.is_frozen is True:
            print(f"---> Unfreezing the backbone {epoch} = {self.frozen_epochs} ...")
            model_utils.unfreeze_parameters(self.model)
            self.is_frozen = False

        elif self.frozen_epochs == 0 and epoch == 0:
            print(f"---> Unfreezing the backbone ...")
            model_utils.unfreeze_parameters(self.model)
            self.is_frozen = False

        return


class NoiseSchedulers:
    def __init__(self, beta_scheduler, beta_1, beta_T, T):
        self.beta_scheduler = beta_scheduler
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T

    def betas_for_alpha_bar(self, max_beta=0.999):
        """
        Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
        (1-beta) over time from t = [0,1].
        Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
        to that part of the diffusion process.
        Args:
            T (`int`): the number of betas to produce.
            max_beta (`float`): the maximum beta to use; use values lower than 1 to
                         prevent singularities.
        Returns:
            betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
        """

        def alpha_bar(time_step):
            return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

        betas = []
        for i in range(self.T):
            t1 = i / self.T
            t2 = (i + 1) / self.T
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return torch.tensor(betas, dtype=torch.float32)

    def scheduler(self):
        if self.beta_scheduler == "linear":
            betas = torch.linspace(self.beta_1, self.beta_T, self.T, dtype=torch.float32)
        elif self.beta_scheduler == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            betas = (torch.linspace(self.beta_1 ** 0.5, self.beta_T ** 0.5, self.T, dtype=torch.float32) ** 2)
        elif self.beta_scheduler == "cosine":
            # Glide cosine schedule
            betas = self.betas_for_alpha_bar()
        elif self.beta_scheduler == "sigmoid":
            temperature = 0.9
            betas = torch.linspace(-6, 6, self.T)
            betas = torch.sigmoid(betas / temperature) * (self.beta_T - self.beta_1) + self.beta_1
        else:
            raise NotImplementedError(f"{self.beta_scheduler} is not implemented ")

        return betas


class EarlyStop:
    """
    Implementation of an early stop criterion

    Args:
    -----
    mode: string ['min', 'max']
        whether we validate based on maximizing or minmizing a metric
    delta: float
        threshold to consider improvements
    use_early_stop: bool
        If True, early stopping functionalities are computed.
    patience: integer
        number of epochs without improvement to trigger early stopping
    """

    def __init__(self, mode="min", delta=1e-6, use_early_stop=True, patience=10):
        """ Early stopper initializer """
        assert mode in ["min", "max"]
        self.mode = mode
        self.delta = delta
        self.use_early_stop = use_early_stop
        self.patience = patience
        self.counter = 0

        if mode == "min":
            self.best = 1e15
            self.criterion = lambda x: x < (self.best - self.delta)
        elif mode == "max":
            self.best = 1e-15
            self.criterion = lambda x: x > (self.best + self.delta)

        return

    def __call__(self, value, epoch=0, writer=None):
        """
        Comparing current metric agains best past results and computing if we
        should early stop or not

        Args:
        -----
        value: float
            validation metric measured by the early stopping criterion
        wirter: TensorboardWriter
            If not None, TensorboardWriter to log the early-stopper counter

        Returns:
        --------
        stop_training: boolean
            If True, we should early stop. Otherwise, metric is still improving
        """
        if not self.use_early_stop:
            return False

        are_we_better = self.criterion(value)
        if are_we_better:
            self.counter = 0
            self.best = value
        else:
            self.counter = self.counter + 1

        stop_training = True if(self.counter >= self.patience) else False
        if stop_training:
            print(f"It has been {self.patience} epochs without improvement")
            print("  --> Trigering early stopping")

        if writer is not None:
            writer.add_scalar(
                name="Learning/Early-Stopping Counter",
                val=self.counter,
                step=epoch + 1,
            )

        return stop_training
    
class ExponentialLRSchedule:
    """
    Exponential LR Scheduler that decreases the learning rate by multiplying it
    by an exponentially decreasing decay factor:
        LR = LR * gamma ^ (step/total_steps)

    Args:
    -----
    optimizer: torch.optim
        Optimizer to schedule
    init_lr: float
        base learning rate to decrease with the exponential scheduler
    gamma: float
        exponential decay factor
    total_steps: int/float
        number of optimization steps to optimize for. Once this is reached,
        lr is not decreased anymore
    """

    def __init__(self, optimizer, init_lr, gamma=0.5, total_steps=100_000):
        """ Module initializer """
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.gamma = gamma
        self.total_steps = total_steps
        return

    def update_lr(self, step):
        """ Computing exponential lr update """
        new_lr = self.init_lr * self.gamma ** (step / self.total_steps)
        return new_lr

    def step(self, iter):
        """ Scheduler step """
        if iter < self.total_steps:
            for params in self.optimizer.param_groups:
                params["lr"] = self.update_lr(iter)
        elif iter == self.total_steps:
            print(f"Finished exponential decay due to reach of {self.total_steps} steps")
        return

    def state_dict(self):
        """ State dictionary """
        state_dict = {key: value for key, value in self.__dict__.items() if key != 'optimizer'}
        return state_dict

    def load_state_dict(self, state_dict):
        """ Loading state dictinary """
        self.init_lr = state_dict["init_lr"]
        self.gamma = state_dict["gamma"]
        self.total_steps = state_dict["total_steps"]
        return
    

class LRWarmUp:
    """
    Class for performing learning rate warm-ups. We increase the learning rate
    during the first few iterations until it reaches the standard LR

    Args:
    -----
    init_lr: float
        initial learning rate
    warmup_steps: integer
        number of optimization steps to warm up for
    max_epochs: integer
        maximum number of epochs to warmup. It overrides 'warmup_step'
    """

    def __init__(self, init_lr, warmup_steps, max_epochs=1):
        """ Initializer """
        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.max_epochs = max_epochs
        self.active = True

    def __call__(self, iter, epoch, optimizer):
        """ Computing actual learning rate and updating optimizer """
        if iter > self.warmup_steps or epoch >= self.max_epochs:
            if self.active:
                self.active = False
                lr = self.init_lr
                print("Finished learning rate warmup period...")
        else:
            lr = self.init_lr * (iter / self.warmup_steps)
            for params in optimizer.param_groups:
                params["lr"] = lr

        return

    def state_dict(self):
        """ State dictionary """
        state_dict = {key: value for key, value in self.__dict__.items()}
        return state_dict

#
