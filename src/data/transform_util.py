
import torch
import numpy as np


def normalization_0_1(rdmaps, epsilon=1e-07):
    # normalize the dataset to [0,1] the with respect to the highest max value and the lowest min value of the whole dataset
    maps_min, maps_max = torch.min(rdmaps), torch.max(rdmaps)

    normalized_rdmaps = (rdmaps - maps_min) / (maps_max - maps_min)
    # add the espilon to the data to avoid having 0.0 in the range-doppler maps; log10(0) = nan
    return normalized_rdmaps + epsilon


def normalize_complex(rdmaps, min_max=(0, 1)):
    """
    Args:
        rdmaps (torch tensor): The complex-valued RD Maps

    Calculate the magnitude and the phase of RD maps. Normalize the magnitudes to [0,1].
    Bring the data to a + jb converting the polar format to cartesian: a = r * cos(theta); b = r * sin(theta)

    Returns:
        torch tensor: Normalized RD Maps according to magnitude only (phase is not changed)
    """
    # obtain magnitude (radius) and phase (angle) of rd maps
    magnitude = torch.abs(rdmaps)
    phase = torch.angle(rdmaps)
    # normalize only the radius (magnitude) while keeping angle (phase) as before
    if min_max[0] == 0:
        normalized_magnitude = normalization_0_1(magnitude)
    elif min_max[0]  == -1:
        normalized_magnitude = magnitude * (min_max[1] - min_max[0]) + min_max[0]

    # create the normalized rd maps using the normalized magnitude; a + jb -->  a = r * cos (angle); b = r * sin(angle)
    a = normalized_magnitude * torch.cos(phase)
    b = normalized_magnitude * torch.sin(phase)
    # in pytorch, one cannot write directly  a + jb; but instead stack a and b along the last dimension and then use torch.view_as_complex()
    rdmaps_stacked = torch.stack([a, b], dim=-1)
    norm_rdmaps = torch.view_as_complex(rdmaps_stacked)

    return norm_rdmaps 


def complex_log_normalize(data, log=True, epsilon=1e-14, device=None): 
    # Auto-detect device if not provided
    if device is None:
        device = data.device
    data = data.to(device)

    magnitude = torch.abs(data)
    # take the logarithm of magnitude; add epsilon since log10(0) -> undefined
    log_magnitude = torch.log10(magnitude+epsilon) if log else magnitude 

    # normalize only the radius (magnitude) while keeping angle (phase) as before
    min_mag, max_mag = torch.min(log_magnitude), torch.max(log_magnitude)
    normalized_log_magnitude = (log_magnitude - min_mag) / (max_mag - min_mag) 

    # Reconstruct complex numbers with preserved phase; use the magnitude instead of magnitude_log since you have data in the nominator; data = magnitude * exp(theta)
    normalized_log_data = normalized_log_magnitude * (data / (magnitude + epsilon))

    return normalized_log_data


def log_abs_and_normalize(data, log_scale=True, device=None):
    """Take the magnitude of the data;
    Take the logarithm of the data if log_scale is True; 
    Then normalize the data to [0,1] 

    Args:
        data (tensor): RD maps
        log_scale (bool): To convert the data into logarithmic scale. Defaults to True.

    Returns:
        _normalized_data: Magnitude of the data in log scale and normalized to [0,1]  
    """
    # Auto-detect device if not provided
    if device is None:
        device = data.device
    data = data.to(device)

    eps = 1e-14
    magnitude_data = torch.abs(data)
    if log_scale:
        magnitude_data = torch.log10(magnitude_data + eps)
    normalized_data_0_1 = normalization_0_1(magnitude_data, epsilon=1e-7)

    return normalized_data_0_1


def abs_and_normalize(data):
    magnitude_data = torch.abs(data)
    log_data = torch.log10(magnitude_data)
    normalized_data_0_1 = normalization_0_1(log_data, epsilon=1e-7)
    return normalized_data_0_1
 

def transform_augment(rdmaps_list, min_max=(-1, 1)):    
    # expects data to be already normalized to [0, 1]
    ret_maps = [map * (min_max[1] - min_max[0]) + min_max[0] for map in rdmaps_list]
    return ret_maps


def transform_complex_augment(rdmaps_list, min_max=(0, 1)):  
    rdmaps =[] 
    for map in rdmaps_list:
        # expects data to be already normalized to [0, 1]
        # normalized_map = normalize_complex(map, min_max)
        normalized_map = complex_log_normalize(map, log=False)
        rdmaps.append(normalized_map)

    return rdmaps