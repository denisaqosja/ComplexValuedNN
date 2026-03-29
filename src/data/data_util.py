import os
import torch 
import pickle
import numpy as np

 
def read_npy_files(path, data_len=-1):
    if data_len < 0:
        filenames = sorted(os.listdir(path))
    else:
        filenames = sorted(os.listdir(path))[:data_len] 

    filenames = sorted(os.listdir(path))
    data_paths = []
    for filename in filenames:
        if ".npy" in filename:
            data_path = os.path.join(path, filename)
            data_paths.append(data_path)
 
    return data_paths


def read_npz_files(path, data_len=-1):
    if data_len < 0:
        npz_files = sorted(os.listdir(path))
    else:
        npz_files = sorted(os.listdir(path))[:data_len] 

    # Precompute mapping from global idx to (file, local idx)
    rdmaps_hr = []
    for i, file in enumerate(npz_files):
        filepath = os.path.join(path, file)
        print(filepath)
        with np.load(filepath) as data:
            keys = list(data.files)
            for key in keys:
                rdmaps_hr.append((filepath, key))

    return rdmaps_hr


def load_npy(path):
    filenames = sorted(os.listdir(path))
    data = []
    for filename in filenames:
        if ".npy" in filename:
            data_f = np.load(os.path.join(path, filename))
            data.append(data_f)
    data = np.stack(data, axis=0)
    return np.squeeze(data)


def load_data(path, data_len, pickle=False):
    if pickle:
        data = load_pickle(path)
    else:
        datafiles_path = read_npy_files(path, data_len)
        data = load_npy(datafiles_path)

    # convert data to tensors: it is better than working with numpy because np.fft casts the data always to complex128 even when it is not wanted
    list_tensors = [torch.from_numpy(rdmap) for rdmap in data]
    torch_data = torch.stack(list_tensors)

    return torch_data

 
def load_pickle(datapath):
    filenames = sorted(os.listdir(datapath))
    for filename in filenames:
        if ".pkl" in filename:
            print(filename)
            with open(os.path.join(datapath, filename),'rb') as f:
                data = pickle.load(f)
    return data
 


