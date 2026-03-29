import torch, os
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml

from torch.utils.data import DataLoader
from data.rdmaps_loader import LRHRDataset_Complex

from model.complex_unet import UNet
from model.tensorboard_writer import TensorboardWriter
from model.setup_model import *


class Evaluator():
    def __init__(self, exp_name, ckpt_name):
        # Load configuration from the experiment path
        self.exp_name = exp_name
        self.exp_path = os.path.join(os.getcwd(), "experiments", exp_name)
        self.params = self.load_config(self.exp_path)

        # Set device: use config value if CUDA is available, otherwise default to CPU
        device_str = self.params['training']['device']
        if device_str == 'cuda' and not torch.cuda.is_available():
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(device_str)
        # data
        self.test_dataloader = self.load_data(self.params["data"])
        # model
        self.setup_model_arch()
        self.load_checkpoint(self.params['paths']["checkpoints"], ckpt_name)


    def load_config(self, exp_path):
        """Load configuration from YAML file."""
        config_path = os.path.join(exp_path, "config.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config

    def load_data(self, data_config):
        test_dataset = LRHRDataset_Complex(data_config, split="test")
        test_loader = DataLoader(test_dataset, 
                                 batch_size=data_config['test']['batch_size'],
                                 shuffle=data_config['val']['use_shuffle'])
        return test_loader
    

    def criterion(self, x, y):
        dist = nn.MSELoss()
        return dist(x, y)
    

    def setup_model_arch(self): 
        # Load UNet config from params
        unet_config = self.params['model']['unet']
        self.model = UNet(in_channel=unet_config['in_channel'], 
                        out_channel=unet_config['out_channel'], 
                        inner_channel=unet_config['inner_channel'], 
                        channel_mults=unet_config['channel_mults'], 
                        attn_res=unet_config['attn_res'], 
                        num_res_blocks=unet_config['num_res_blocks'], 
                        dropout=unet_config['dropout']
                    ).to(self.device)
    
        self.writer = TensorboardWriter(logdir=self.params['paths']['tb_logger']) 
        return

    def test(self):
        print(f"Testing the model ...")
        self.model.eval()
        with torch.no_grad():
            progress_bar = tqdm(enumerate(self.test_dataloader), total=len(self.test_dataloader))
            for iter, test_data in progress_bar:
                x = test_data['HR'].to(self.device)
                recon = self.model(x)
                real_V_loss = self.criterion(recon.real, x.real)
                complex_v_loss = self.criterion(recon.imag, x.imag)
                loss = real_V_loss + complex_v_loss 
            
                # complex correlation coefficient
                rho = torch.sum(x * torch.conj(recon)) / torch.sqrt(torch.sum(torch.abs(x)**2) * torch.sum(torch.abs(recon)**2))

                self.writer.add_scalar(
                            name="Test/Loss", 
                            val=loss.item(), 
                            step=iter 
                        )
                self.log_tensorboard(x.cpu(), recon.cpu(), rho, iter, split="Test")
          
        return 

    def log_tensorboard(self, x, recon, rho, global_iter, split="Train"):
        magnitude_diff = torch.abs(x) - torch.abs(recon)
        phase_x = torch.angle(x)  
        phase_recon = torch.angle(recon)
        phase_diff = np.angle(np.exp(1j * (phase_x - phase_recon).detach().cpu().numpy()))

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
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print(f"Checkpoint is loaded!")
        return 

    def get_velocity_range_bins(self):
        # for measured data, the range and velocity are as follows:
        N_velocity_bins = 256
        v_max = 25.90850197 
        v_min = -1 * v_max

        N_range_bins = 128
        r_max = 237.96026354
        r_min = 0

        # create array of velocity bins
        velocity_bins = np.linspace(v_min, v_max, N_velocity_bins)  
        range_bins = np.linspace(r_min, r_max, N_range_bins)

        return velocity_bins, range_bins

    def show_reconstructions(self):
        self.model.eval()
        test_data = next(iter(self.test_dataloader))
        imgs = test_data['HR'].to(self.device)

        # get the values for x-axis and y-axis
        velocity_bins, range_bins = self.get_velocity_range_bins()
  
        with torch.no_grad():
            recon = self.model(imgs)
        
        imgs = imgs.cpu().numpy()
        recon = recon.cpu().numpy()

        fig, axes = plt.subplots(2, 4, figsize=(12, 5), constrained_layout=True)
        # --- Plot originals (top row) ---
        for i in range(4):
            ax = axes[0, i]
            im = ax.pcolor(velocity_bins, range_bins, np.abs(imgs[i, 0]))
            ax.set_title(f"Orig. Magnitude Map {i+1}", fontsize=12, fontweight='bold')
            # ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # --- Plot reconstructions (bottom row) ---
        for i in range(4):
            ax = axes[1, i]
            im = ax.pcolor(velocity_bins, range_bins, np.abs(recon[i, 0]))
            ax.set_title(f"Recon. Magnitude {i+1}", fontsize=12, fontweight='bold')
            # ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.savefig(os.path.join(self.params['paths']['results'], "magnitude_imgs.png"), dpi=150)
        plt.show()
        plt.close(fig)
        

if __name__=="__main__":
    # data 
    torch.manual_seed(0)
    # model
    exp_name = "exp_complex-valued-UNet_2025-11-13_08-55-14"  
    ckpt = "epoch_100.pth"
    trainer = Evaluator(exp_name, ckpt)
    trainer.test()
    trainer.show_reconstructions()
