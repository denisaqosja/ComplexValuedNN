import torch, os
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from datetime import datetime
import matplotlib.pyplot as plt
import yaml

from torch.utils.data import DataLoader
from data.rdmaps_loader import LRHRDataset_Complex

from model.complex_unet import UNet
from model.tensorboard_writer import TensorboardWriter
from model.schedulers import GradualWarmupScheduler
from model.setup_model import *


class Trainer():
    def __init__(self):
        # Load configuration from YAML
        config_path = os.path.join(os.getcwd(), "configs", "cvnn.yaml")
        self.params = self.load_config(config_path)
        
        # create experiment directories
        self.create_exp_directories()
        # Set device: use config value if CUDA is available, otherwise default to CPU
        device_str = self.params['training']['device']
        if device_str == 'cuda' and not torch.cuda.is_available():
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(device_str)

        # Save the config file to the experiment directory
        self.save_config_to_exp_dir()
        
        # Load training parameters from config
        self.n_epochs = self.params['training']['n_epochs']
        # load data
        self.train_dataloader, self.val_loader = self.load_data(self.params["data"])
        # setup model
        self.setup_model_arch()
        
        
    def save_config_to_exp_dir(self):
        """Save the config file to the experiment directory for reproducibility."""
        config_save_path = os.path.join(os.getcwd(), self.paths["exp_path"], "config.yaml")
        with open(config_save_path, 'w') as f:
            yaml.dump(self.params, f, default_flow_style=False, sort_keys=False)
        return

    
    def load_config(self, config_path):
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config


    def create_exp_directories(self):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.exp_name = self.params['training']['exp_name']

        self.paths = {
            "exp_path": f"experiments/{self.exp_name}_{timestamp}",
            "tb_logger": f"experiments/{self.exp_name}_{timestamp}/tb_logger",
            "results": f"experiments/{self.exp_name}_{timestamp}/results",
            "checkpoints": f"experiments/{self.exp_name}_{timestamp}/checkpoints"
        }

        for _, (_, values) in enumerate(self.paths.items()):
            if not os.path.exists(os.path.join(os.getcwd(), values)):
                os.makedirs(os.path.join(os.getcwd(), values))
        
        # Add paths to the config, needed when evaluating
        self.params['paths'] = self.paths
        
        return 
    

    def load_data(self, data_config):
        train_dataset = LRHRDataset_Complex(data_config, split="train")
            
        train_loader = DataLoader(train_dataset, 
                                  batch_size=data_config['train']['batch_size'], 
                                  shuffle=data_config['train']['use_shuffle'])

        val_dataset = LRHRDataset_Complex(data_config, split="val")
        val_loader = DataLoader(val_dataset, 
                                batch_size=data_config['val']['batch_size'], 
                                shuffle=data_config['val']['use_shuffle'])

        return train_loader, val_loader


    def criterion(self, x, y):
        return self.mse_loss(x, y)


    def setup_model_arch(self): 
        # Load UNet config from params
        unet_config = self.params['model']['unet']
        # load the UNet architecture with the specified parameters
        self.model = UNet(in_channel=unet_config['in_channel'], 
                          out_channel=unet_config['out_channel'], 
                          inner_channel=unet_config['inner_channel'], 
                          channel_mults=unet_config['channel_mults'], 
                          attn_res=unet_config['attn_res'], 
                          num_res_blocks=unet_config['num_res_blocks'], 
                          dropout=unet_config['dropout']
                    ).to(self.device)
        
        # define the loss function 
        self.mse_loss = nn.MSELoss()
        
        # define the optimizer
        optimizer_config = self.params['optimizer']
        self.optimizer = optim.AdamW(self.model.parameters(), 
                                     lr=optimizer_config['lr'], 
                                     weight_decay=optimizer_config['weight_decay'])
        # calculate total training steps
        num_batches_per_epoch = len(self.train_dataloader)
        self.totalN_training_steps = self.n_epochs * num_batches_per_epoch
        # calculate warmp-up steps
        self.total_warmup_steps = 0.05 * self.totalN_training_steps  # 5% warmup
        print(f"Total training steps: {self.totalN_training_steps = }")
        print(f"Total warmup steps: {self.total_warmup_steps = }")

        self.cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,
                                                                    T_max=self.totalN_training_steps, 
                                                                    eta_min=0, last_epoch=-1)
        self.warmUp_scheduler = GradualWarmupScheduler(optimizer=self.optimizer,
                                                      multiplier=optimizer_config['warmup']['multiplier'], 
                                                      warm_epoch=self.total_warmup_steps,
                                                      start_lr = optimizer_config["lr"] * 1e-4,
                                                      after_scheduler=self.cosine_scheduler)
        self.writer = TensorboardWriter(logdir=os.path.join(os.getcwd(), self.paths["tb_logger"]))
    
        return


    def train(self):
        for epoch in range(self.n_epochs):
            # train
            self.model.train()
            running_loss = self.train_epoch(epoch)
            epoch_loss = running_loss / len(self.train_data)
            print(f"Training {epoch_loss = }")
            # log loss to tensorboard
            self.writer.add_scalar(
                        name="Train/EpochLoss", 
                        val=epoch_loss, 
                        step=epoch 
                    )
           
            # save the checkpoint of a complex model...
            if epoch % 5 ==0:
                savepath = os.path.join(os.getcwd(), self.paths['checkpoints'], f"epoch_{epoch}.pth")
                # """
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict()
                    },
                    savepath
                )
            # validate
            #if epoch % 5==0:
            self.epoch_validation(epoch)

        savepath = os.path.join(os.getcwd(), self.paths['checkpoints'], f"epoch_{epoch}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
            },
            savepath
        )
        print("Validating the final epoch ...")
        self.model.eval()
        return 


    def train_epoch(self, epoch):
        print(f"Training the model - Epoch {epoch} ...")
        running_loss = 0.0
        self.model.train()
        progress_bar = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader))
        for iter, train_data in progress_bar:
            global_iter = iter + epoch * len(self.train_dataloader)
           
            # get the reconstruction from the model
            x = train_data['HR'].to(self.device)
            self.optimizer.zero_grad()
            out = self.model(x)
           
            # calculate the loss
            real_V_loss = self.criterion(out.real, x.real)
            complex_v_loss = self.criterion(out.imag, x.imag)
            loss = real_V_loss + complex_v_loss 

            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * x.size(0)
            self.warmUp_scheduler.step()

            # complex correlation coefficient
            rho = torch.sum(x * torch.conj(out)) / torch.sqrt(torch.sum(torch.abs(x)**2) * torch.sum(torch.abs(out)**2))

            self.writer.add_scalar(
                        name="Train/StepLoss", 
                        val=loss.item(), 
                        step=global_iter 
                    )
            
            self.writer.add_scalar(
                    name="Train/LR", 
                    val=self.optimizer.param_groups[0]["lr"], 
                    step=global_iter 
                )
    
            self.log_tensorboard(x, out, rho, global_iter)

            # remove for complex data
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Plots histograms of complex-valued weight distributions for each layer
            # self.histogram()

        return running_loss 
    

    def epoch_validation(self, epoch):
        print(f"Validating the model - Epoch {epoch} ...")
        self.model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            progress_bar = tqdm(enumerate(self.val_loader), total=len(self.val_loader))
            for i, val_data in progress_bar:
                global_iter = i + epoch * len(self.train_dataloader.dataset)
                
                # get the reconstruction from the model
                x = val_data['HR'].to(self.device)
                recon = self.model(x) 
                
                # calculate the loss
                real_V_loss = self.criterion(x.real, recon.real)
                complex_v_loss = self.criterion(x.imag, recon.imag)
                loss = real_V_loss + complex_v_loss 
                eval_loss += loss.item() * x.size(0)
            
                # complex correlation coefficient
                rho = torch.sum(x * torch.conj(recon)) / torch.sqrt(torch.sum(torch.abs(x)**2) * torch.sum(torch.abs(recon)**2))
                self.writer.add_scalar(
                        name="Val/StepLoss", 
                        val=loss.item(), 
                        step=global_iter 
                    )
                self.log_tensorboard(x.cpu(), recon.cpu(), rho, global_iter, split="Val")

            val_loss = eval_loss / len(self.val_loader.dataset)
            print(f"Validation loss: {val_loss}")
            # log loss to tensorboard
            self.writer.add_scalar(
                    name="Val/EpochLoss", 
                    val=val_loss, 
                    step=epoch 
                )
            # log into tensorboard magnitude of rdmaps; Batch GPU→CPU transfer once, not separately
            x_cpu = x[:4].cpu()
            recon_cpu = recon[:4].cpu()
            hr_sr_tensor = torch.stack((abs(x_cpu), abs(recon_cpu)), dim=1)
            self.writer.add_grid_rdmaps(images=hr_sr_tensor, n_rows=2, n_cols=4, step=epoch, tag="Valid/RDMaps")
            
        self.model.train()
        return 

    def log_tensorboard(self, x, recon, rho, global_iter, split="Train"):
        magnitude_diff = torch.abs(x) - torch.abs(recon)
        phase_x = torch.angle(x)  
        phase_recon = torch.angle(recon)
        # phase_diff = np.angle(np.exp(1j * (phase_x - phase_recon).detach().cpu().numpy()))
        phase_diff = torch.angle(torch.exp(1j * (phase_x - phase_recon).detach()))

        magnitude_rho = torch.abs(rho)
        phase_offset_rho = torch.angle(rho)

        # log scalar
        self.writer.add_scalar(
                    name=f"{split}/MagnitudeDifference/Mean", 
                    val=magnitude_diff.mean(), 
                    step=global_iter 
                )
        self.writer.add_scalar(
                    name=f"{split}/MagnitudeDifference/St.deviation", 
                    val=magnitude_diff.std(), 
                    step=global_iter 
                )
        self.writer.add_scalar(
                    name=f"{split}/MagnitudeDifference/Rho", 
                    val=magnitude_rho, 
                    step=global_iter 
                )
        self.writer.add_scalar(
                    name=f"{split}/PhaseDifference/Mean", 
                    val=phase_diff.mean(), 
                    step=global_iter 
                )
        self.writer.add_scalar(
                    name=f"{split}/PhaseDifference/St.deviation", 
                    val=phase_diff.std(), 
                    step=global_iter 
                )
        self.writer.add_scalar(
                    name=f"{split}/PhaseDifference/Rho", 
                    val=phase_offset_rho, 
                    step=global_iter 
                )
        return 
    
    def load_checkpoint(self, ckpt_path, ckpt):
        checkpoint_path = os.path.join(os.getcwd(), ckpt_path, ckpt)
        print(f"{checkpoint_path = }")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Checkpoint is loaded!")
        return 
        # print(f"The state dict from the saved checkpoint: {checkpoint['model_state_dict']}")

    def show_reconstructions(self):
        self.model.eval()
        data = next(iter(self.val_loader))
        imgs = data['HR'].to(self.device)
        t = torch.randint(self.T, size=(imgs.shape[0], ), device=imgs.device)
        with torch.no_grad():
            recon = self.model(imgs, t)
        
        title = "Validation Samples"
        magnitude_diff = torch.abs(torch.abs(imgs) - torch.abs(recon))
        phase_diff = torch.angle(imgs) - torch.angle(recon)  # torch.abs(torch.angle(imgs) - torch.angle(recon))
        print(f"{magnitude_diff.mean() = }")
        print(f"{phase_diff.mean() = }")
        print(f"{abs(recon).min() = }, {abs(recon).max() = }, {recon.dtype}")
        print(f"{torch.mean(recon) = }")

        imgs = imgs.cpu().numpy()
        recon = recon.cpu().numpy()

        print(f"{imgs.shape = }")
        fig, axes = plt.subplots(4, 2, figsize=(12, 3))
        for i in range(2):
            axes[0, i].imshow(abs(imgs[i, 0]), vmax=1, vmin=1e-12)
            #axes[0, i].imshow(imgs[i, 0], vmax=1, vmin=1e-12)
            axes[0, i].axis('off')
            axes[0, i].set_title("Orig. Magnitude RDMap")
            axes[1, i].imshow(abs(recon[i, 0]), vmax=1, vmin=1e-12)
            #axes[1, i].imshow(abs(recon[i, 0]), vmax=1, vmin=1e-8)
            axes[1, i].axis('off')
            axes[1, i].set_title("Recon. RDMap")
            axes[2, i].imshow(np.angle(imgs[i, 0]))
            axes[2, i].axis('off')
            axes[2, i].set_title("Orig. Phase RDMap")
            axes[3, i].imshow(np.angle(recon[i, 0]))
            # axes[3, i].colorbar()
            axes[3, i].axis('off')
            axes[3, i].set_title("Recon. Phase RDMap")
        plt.suptitle(title)
        plt.show()


    def histogram(self):
        # Plots histograms of complex-valued weight distributions for each layer
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if torch.is_complex(param):
                    print(name, "Real mean:", param.grad.real.mean().item())
                    print(name, "Imag mean:", param.grad.imag.mean().item())
                else:
                    print(f"{name} is real")
                    print("Grad mean:", param.grad.mean().item())

                if torch.is_complex(param):
                    real_vals, imag_vals, abs_vals = [], [], []
                    real_vals.append(param.data.real.cpu().flatten())
                    imag_vals.append(param.data.imag.cpu().flatten())

                    real_vals_ = torch.cat(real_vals)
                    imag_vals_ = torch.cat(imag_vals)
                    # Combine real + imag into one 1D tensor
                    combined_vals = torch.cat([real_vals_, imag_vals_])

                    # get the module name
                    module_name = ".".join(name.split(".")[:-1])
                    param_name = name.split(".")[-1]
                    module = dict(self.model.named_modules()).get(module_name, None)
                    module_type = module.__class__.__name__ if module is not None else "N/A"
                    # plot the histogram of the weights 

                    plt.hist(combined_vals.numpy(), bins=20)
                    plt.title(f"Histogram of real + imaginary weight values {module_type} {param_name}")
                    plt.xlabel("Value")
                    plt.ylabel("Count")
                    plt.show()


if __name__=="__main__":
    # data 
    torch.manual_seed(0)
    # model
    trainer = Trainer()
    trainer.train()
    trainer.show_reconstructions()

#