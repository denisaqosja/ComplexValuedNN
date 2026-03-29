import torch 
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

from data import data_util, transform_util, helpers
from data.processing_util import reduce_resolution, zeropadding_rdmap, preprocess_data


class BaseDataset(Dataset):
    def __init__(self, dataset_opt, split, need_LR=False):
        super().__init__()

        """Create the dataset for feeding into the NN
    
        Args:
            dataroot        : directory where the dataset is located
            datatype        : whether the data samples are rdmaps (.npy or mat) or images
            do_sr_range     : (bool) whether to apply SR on range
            do_sr_doppler   : (bool) whether to apply SR on doppler
            hr_range_bins   : the spatial dimension accross range,. e.g 128
            hr_doppler_bins : the spatial dimension accross doppler,. e.g 256
            split           : train or test. Defaults to "train".
        """
        dataroot=dataset_opt['dataroot']
        self.k=dataset_opt['resolving_factor_k']
        self.do_sr_range=dataset_opt['do_sr_range']
        self.do_sr_doppler=dataset_opt['do_sr_doppler']
        self.N_fft_fast=dataset_opt['range_bins']
        self.N_fft_slow=dataset_opt['doppler_bins']
        self.undo_fft = dataset_opt['is_data_in_fft_domain']
        self.windowing = dataset_opt['apply_window']
        self.window_type = dataset_opt['window_type']
        self.complex_valued=dataset_opt['complex_data']
        data_len=dataset_opt[split]['data_len']
        
        self.need_LR = need_LR
        # Set device for preprocessing - use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
        self.get_system_parameters(dataset_opt)

        print(f"{dataroot = }")
        self.npz = True if dataset_opt["datatype"] == "npz" else False
        self.train_ADC = dataset_opt["train_ADC_signal"]

        if self.npz: # the simulated data is saved as npz
            self.hr_data_files = data_util.read_npz_files(f'{dataroot}/HR/{split}', data_len=data_len)
        else: # the measured data is saved as npy: multiple arrays into 1 file
            data_path = f'{dataroot}/HR/{split}'
            # self.hr_data_files = data_util.read_npy_files(data_path, data_len=data_len)
            
            self.hr_data_files = np.squeeze(data_util.load_npy(data_path))

        print(f"{dataroot = }, {split = } ")
        print(f"Data preprocessing device: {self.device}")
        return 
    
    def get_system_parameters(self, dataset_opt):
        params = dataset_opt["system_parameters"]
        self.wavelength = params["c"] / params["fc"]
        # fmcw system
        self.PRF = 1/params["chirpLen_s"]

        range_bins_vec, _ = helpers.get_range_bin(params["bandwith"], params["c"], self.N_fft_fast)
        self.range_res = range_bins_vec[1] - range_bins_vec[0]

        vel_bins_vec = helpers.get_velocity_bin(self.N_fft_slow, self.PRF, self.wavelength)
        self.doppler_res = vel_bins_vec[1] - vel_bins_vec[0]

        return

    def get_SR(self, hr_data):
        hr_data = hr_data.to(self.device)
        # assert hr_data.device.type == 'cuda', f"Expected data on CUDA but got {hr_data.device}"
        
        # reduce the resolution of HR maps, and obtain the LR and SR versions
        lr_data = reduce_resolution(hr_data, velocity=self.do_sr_doppler, range=self.do_sr_range, k=self.k, undo_fft=self.undo_fft)   
        sr_data = zeropadding_rdmap(lr_data, velocity=self.do_sr_doppler, range=self.do_sr_range, k=self.k, 
                                    sr_range=self.N_fft_fast, sr_velocity=self.N_fft_slow, undo_fft=self.undo_fft)   
        
        return lr_data, sr_data

    def do_preprocessing(self, hr_data, lr_data, sr_data):
        # process the HR data to do windowing ; can't do the processing before obtaining the LR/SR versions
        processed_hr_data = preprocess_data(hr_data, self.window_type, self.windowing, self.device)
        # process the LR & SR data to do windowing 
        processed_sr_data = preprocess_data(sr_data, self.window_type, self.windowing, self.device)
        processed_lr_data = preprocess_data(lr_data, self.window_type, self.windowing, self.device) if self.need_LR else None
            
        return processed_hr_data, processed_lr_data, processed_sr_data

    def abs_norm(self, hr_data, lr_data, sr_data):
        # process the HR data to do normalization
        hr_data = torch.squeeze(transform_util.log_abs_and_normalize(hr_data), dim=0)
        # process the LR & SR data to do normalization
        sr_data = torch.squeeze(transform_util.log_abs_and_normalize(sr_data), dim=0)
        lr_data = torch.squeeze(transform_util.log_abs_and_normalize(lr_data), dim=0) if self.need_LR else None
           
        return hr_data, lr_data, sr_data
    
    
    def complex_abs_norm(self, hr_data, lr_data, sr_data):  
        # process the HR data to do normalization
        hr_data = torch.squeeze(transform_util.complex_log_normalize(hr_data), dim=0)
        # process the LR & SR data to do normalization
        sr_data = torch.squeeze(transform_util.complex_log_normalize(sr_data), dim=0)
        lr_data = torch.squeeze(transform_util.complex_log_normalize(lr_data), dim=0) if self.need_LR else None
           
        return hr_data, lr_data, sr_data


    def __getitem__(self, idx):
        raise NotImplementedError("Base class does not implement __getitem__")
    
    def __len__(self):
        return len(self.hr_data_files)


class LRHRDataset_Measured(BaseDataset):
    def __init__(self, dataset_opt, split="train", need_LR=False):
        super().__init__(dataset_opt, split=split, need_LR=need_LR)
        print(f"Number of data samples: {len(self.hr_data_files)}")

    def process_data(self, hr_data, lr_data, sr_data):
        # process the HR data to do windowing ; can't do the processing before obtaining the LR/SR versions
        processed_hr_data, processed_lr_data, processed_sr_data = self.do_preprocessing(hr_data, lr_data, sr_data)
        
        # process the HR data to do normalization
        self.hr_data, self.lr_data, self.sr_data = self.abs_norm(processed_hr_data, processed_lr_data, processed_sr_data) 
        
        return 

    def __getitem__(self, idx):
        if self.npz:
            file_path, key = self.hr_data_files[idx]
            data = np.load(file_path)
            hr_data = data[key]
        else:
            hr_data = self.hr_data_files[idx]

        hr_data = torch.from_numpy(hr_data).unsqueeze(dim=0)
        hr_data = hr_data.to(self.device)
        
        # do augmentations on the HR sample (e.g. Doppler shift) before getting the SR map 
        # otherwise the SR map has a bright reflection at the position where the zero-Doppler line was removed
        # hr_data = self.do_augmentations(hr_data)  if self.transforms else hr_data

        # Obtaining SR data ...
        lr_data, sr_data = self.get_SR(hr_data)
      
        # Processing all the data for windowing / normalization 
        self.process_data(hr_data, lr_data, sr_data) 
        rdmap_HR = self.hr_data.float()
        rdmap_SR = self.sr_data.float()

        # normalize to [-1, 1]
        if self.need_LR:
            rdmap_LR = self.lr_data.float()
            [rdmap_LR, rdmap_SR, rdmap_HR] = transform_util.transform_augment([rdmap_LR, rdmap_SR, rdmap_HR], min_max=(-1, 1))
            return {'LR': rdmap_LR, 'HR': rdmap_HR, 'SR': rdmap_SR, 'Index': idx}
        else:
            [rdmap_SR, rdmap_HR] = transform_util.transform_augment([rdmap_SR, rdmap_HR], min_max=(0, 1))
            return {'HR': rdmap_HR, 'SR': rdmap_SR, 'Index': idx}
           
    def __len__(self):
        return len(self.hr_data_files)
   


class LRHRDataset_Complex(BaseDataset):
    """
    For now works only for the measured data....
    """
    def __init__(self, dataset_opt, split="train", need_LR=False):
        super().__init__(dataset_opt, split=split, need_LR=need_LR)

        print(f"Number of data samples: {len(self.hr_data_files)}")

    def process_data(self, hr_data, lr_data, sr_data):
        # process the HR data to do windowing ; can't do the processing before obtaining the LR/SR versions
        processed_hr_data, processed_lr_data, processed_sr_data = self.do_preprocessing(hr_data, lr_data, sr_data)

        # process the HR data to do normalization
        self.hr_data, self.lr_data, self.sr_data = self.complex_abs_norm(processed_hr_data, processed_lr_data, processed_sr_data) 
       
        return 

    def __getitem__(self, idx):
        if self.npz:
            file_path, key = self.hr_data_files[idx]
            data = np.load(file_path)
            hr_data = data[key]
        else:
            hr_data = self.hr_data_files[idx]

        hr_data = torch.from_numpy(hr_data).unsqueeze(dim=0)
        hr_data = hr_data.to(self.device)
        
        # Obtaining SR data ...
        lr_data, sr_data = self.get_SR(hr_data)
      
        # Processing all the data for windowing / normalization 
        self.process_data(hr_data, lr_data, sr_data) 
        # Move back to CPU for model input
        rdmap_HR = self.hr_data.cpu().to(torch.complex64)
        rdmap_SR = self.sr_data.cpu().to(torch.complex64)

        # normalize to [-1, 1]
        if self.need_LR:
            # TODO: check if the conversion to cpu is needed
            rdmap_LR = self.lr_data.cpu().to(torch.complex64)
            return {'LR': rdmap_LR, 'HR': rdmap_HR, 'SR': rdmap_SR, 'Index': idx}
        else:
            return {'HR': rdmap_HR, 'SR': rdmap_SR, 'Index': idx}
           
    def __len__(self):
        return len(self.hr_data_files)

#