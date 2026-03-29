
import torch 
import numpy as np

from data.data_util import load_data


def apply_window_numpy(rdmap_orig, window_type="blackman"):
    """Perform windowing on a RD Map
    First revert the fft performed on the map, apply the window, the convert to frequency domain via fft
    The main code is designed for a map of shape = [batch_size, n_range_bins, n_doppler_bins, n_angle_bins].
    If the map that is fed here is of shape = [n_range_bins, n_doppler_bins, n_angle_bins], add a single dim=1 for the batch,
    and remove the single dim before returning the windowed map.
    """
   
    window_fn = {
        "blackman": np.blackman,
        "hamming": np.hamming,
        "hanning": np.hanning
    }
   
    if len(rdmap_orig.shape) == 3:
       rdmap = rdmap_orig[np.newaxis, :, :, :]
    elif len(rdmap_orig.shape) == 4:
        rdmap = rdmap_orig
 
    # design the window for fast-time (range)
    win_range = window_fn[window_type](rdmap.shape[1]).reshape(1, rdmap.shape[1], 1, 1)
    # design the window for slow-time (velocity)
    win_velocity = window_fn[window_type](rdmap.shape[2]).reshape(1, 1, rdmap.shape[2], 1)
   
    # reverse the fft
    rdmap_time = np.fft.fft(np.fft.ifft(np.fft.ifftshift(rdmap, axes=2), axis=2, norm='ortho'), axis=1, norm='ortho')
 
    # multiply rdmap with window on the range and velocity dimensions
    rdmap_time = rdmap_time * win_range
    rdmap_time_win = rdmap_time * win_velocity
 
    # back to the frequency domain
    rdmap_freq_win = np.fft.fftshift((np.fft.fft(np.fft.ifft(rdmap_time_win, axis=1, norm='ortho'), axis=2, norm='ortho')), axes=2)
   
    if len(rdmap_orig.shape) == 3:
        # remove the first dimension added to
        rdmap_freq_win = rdmap_freq_win[0]
    return rdmap_freq_win
 
 
def apply_window(rdmap_orig, window_type="blackman", device=None):
    """Perform windowing on a RD Map
    First revert the fft performed on the map, apply the window, the convert to frequency domain via fft
    The main code is designed for a map of shape = [batch_size, n_range_bins, n_doppler_bins, n_angle_bins].
    If the map that is fed here is of shape = [n_range_bins, n_doppler_bins, n_angle_bins], add a single dim=1 for the batch,
    and remove the single dim before returning the windowed map.
   
    Difference from numpy: np.fft casts the data always to complex128 even when it is not wanted. In this case, to feed it
    to the network complex64 is preferred!
    """
    # Auto-detect device if not provided
    if device is None:
        device = rdmap_orig.device
   
    window_fn = {
            "blackman": torch.blackman_window,
            "hamming": torch.hamming_window,
            "hanning": torch.hann_window
        }
   
    if len(rdmap_orig.shape) == 3:
       rdmap = torch.unsqueeze(rdmap_orig, dim=0)
    elif len(rdmap_orig.shape) == 4:
        rdmap = rdmap_orig
 
    # design the window for fast-time (range)
    win_range = window_fn[window_type](rdmap.shape[1], device=device).reshape(1, rdmap.shape[1], 1, 1)
    # design the window for slow-time (velocity)
    win_velocity = window_fn[window_type](rdmap.shape[2], device=device).reshape(1, 1, rdmap.shape[2], 1)
   
    # reverse the fft
    #rdmap_time = torch.fft.fft(torch.fft.ifft(torch.fft.ifftshift(rdmap, dim=2), dim=2, norm='ortho'), dim=1, norm='ortho')
    rdmap_time = undo_fft_RangeDoppler(rdmap)

    # multiply rdmap with window on the range and velocity dimensions
    rdmap_time = rdmap_time * win_range
    rdmap_time_win = rdmap_time * win_velocity
 
    # back to the frequency domain
    #rdmap_freq_win = torch.fft.fftshift((torch.fft.fft(torch.fft.ifft(rdmap_time_win, dim=1, norm='ortho'), dim=2, norm='ortho')), dim=2)
    rdmap_freq_win = fft_RangeDoppler(rdmap_time_win)

    if len(rdmap_orig.shape) == 3:
        # remove the first dimension added to
        rdmap_freq_win = rdmap_freq_win[0]
    return rdmap_freq_win
   


def preprocess_data(data, window_type="blackman", windowing=True, join_angle_bins=True, epsilon=1e-14, device=None):
    """Process the data before feeding it to the network
    Data comes of shape: n_maps, n_range_bins, n_velocity_bins, n_angle_bins
 
    Args:
        data: range doppler maps
        window_type: the type of the window for removing sidelobes
        windowing: whether to apply the window or not
        join_angle_bins: Defaults to True.
        device: torch device to run operations on (auto-detected if None)
 
    Returns:
        data: processed data
    """
    # Auto-detect device if not provided
    if device is None:
        device = data.device
        
    if windowing:
        data = apply_window(data, window_type=window_type, device=device)
        #print(f"After widnowing: {torch.min(torch.abs(data)) = }, {torch.max(torch.abs(data)) = }")
    if join_angle_bins and  not windowing:
        #warnings.warn("Warning... Cannot join the azimuth/angle maps if windowing is false. Otherwise sidelobes are present ... " )
        #warnings.warn("Warning... Setting join_angles_maps to False...")
        join_angle_bins = False

        # for now, for simplicity and less training time; keep only the RD map from 1st angle
        data = data[:, :, :, 0] 
        # data = data[np.newaxis, :, :, :]
        data = torch.unsqueeze(data, dim=0)

    if join_angle_bins:
        assert len(data.shape) == 3 or len(data.shape) == 4, "The shape of the RD map should be either of dim=3 or dim=4 ..."
 
        data = (data[..., 0] * torch.conj(data[..., 1]))
        if len(data.shape) == 2:
            data = torch.unsqueeze(data, dim=0)
        elif len(data.shape) == 3:
            data = torch.unsqueeze(data, dim=1)
        else:
            raise ValueError("The shape of the RD map after joining angles should be either of dim=3 or dim=4 ...")
        #print(f"After angles join: {torch.min(torch.abs(data)) = }, {torch.max(torch.abs(data)) = }")
 
    return data + epsilon
 

def load_preprocess_data(path, windowing, window_type, pickle=False, join_angle_bins=True, data_len=-1):    
    data = load_data(path, data_len, pickle)
    processed_data = preprocess_data(data, window_type, windowing, join_angle_bins=join_angle_bins)
   
    return processed_data



def get_middle_part_rdmap(bins, lr_reduction_coeff):
        
    middle_point = bins // 2
    # keep only the most middle part of the samples 
    lr_map_min_shape = middle_point - middle_point/lr_reduction_coeff
    lr_map_max_shape = middle_point + middle_point/lr_reduction_coeff
    
    return lr_map_min_shape, lr_map_max_shape


def undo_fft_RangeDoppler(rdmaps, device=None):
    """
    Reverse the fft taken on the Doppler axis. The input data should be of shape (n_maps, n_range_bins, n_doppler_bins, n_angle_bins).
    ***     For applying windows, both the ifft and fft are needed to bring the data back to the raw form (RCS dependent).
            For super-resolution on velocity, only the ifft is sufficient; but to make the function more general, keep the line with 
            both fft and ifft. 
    ***
    Args:
        rdmaps (torch tensor): range-doppler maps with Range in time domain and Doppler in the frequency (Fourier) domain. For performing actions like, 
        windowing, reducing resolution, ect., the Doppler should be in the time domain
        device: torch device for operations (auto-detected from rdmaps if None)
    Returns:
        rdmap_freq (torch tensor): range-dopppler maps with both Range and Doppler in time domain.
    """
    # Auto-detect device if not provided
    if device is None:
        device = rdmaps.device
    rdmaps = rdmaps.to(device)
    
    # reverse the fft - FFT operations run on GPU if device is cuda
    rdmaps_time = torch.fft.fft(torch.fft.ifft(torch.fft.ifftshift(rdmaps, dim=2), dim=2, norm='ortho'), dim=1, norm='ortho')
    # rdmaps_time = torch.fft.ifft(torch.fft.ifftshift(rdmaps, dim=2), dim=2)
    # rdmap_time = np.fft.ifft(np.fft.ifftshift(rdmap, axes=2), axis=2)  # from Kilian 
    return rdmaps_time


def fft_RangeDoppler(rdmaps, device=None):
    """
    Bring Doppler back to the frequency domain. The input data should be of shape (n_maps, n_range_bins, n_doppler_bins, n_angle_bins)
    Args:
        rdmaps (torch tensor): range-doppler maps with Doppler and Range being in time domain. For visualizing range-doppler maps, 
        Doppler should be in the freq domain (Fourier)
        device: torch device for operations (auto-detected from rdmaps if None)
    Returns:
        rdmap_freq (torch tensor): range-dopppler maps with Range in time domain and Doppler in the frequency (Fourier) domain.
    """
    # Auto-detect device if not provided
    if device is None:
        device = rdmaps.device
    
    rdmaps = rdmaps.to(device)
    
    # FFT operations run on GPU if device is cuda
    rdmaps_freq = torch.fft.fftshift((torch.fft.fft(torch.fft.ifft(rdmaps, dim=1, norm='ortho'), dim=2, norm='ortho')), dim=2) 
    # rdmap_freq = torch.fft.fftshift(torch.fft.fft(rdmaps, dim=2), dim=2)
    # rdmap_freq = np.fft.fftshift(np.fft.fft(rdmap_time_win, axis=2, norm='ortho'), axes=2)   # from Kilian 
    return rdmaps_freq



def reduce_resolution(rdmaps, velocity=True, range=True, k=2, undo_fft=False, device=None):
    """
    Reduce the velocity resolution of a range doppler map, by removing samples:
        -  at the beginning and end of the doppler profile in time domain
        -  at the end of the range spectrum (range profile in freq domain)
    The input should be of shape (n_maps, n_range_cells, n_doppler_cells, n_angle_cells)

    Args:
        rdmaps (tensor) : the high-resolution RR Maps
        velocity (bool) : whether to reduce the resolution accross velocity axis. Defaults to True
        range (bool)    : whether to reduce the resolution accross range axis. Defaults to True
        k (int)         : The reduction coefficient of lowering the resolution. Defaults to 2
        undo_fft (bool) : whether to undo the fft on the range-doppler maps
        device          : torch device to run operations on (auto-detected if None)
    Returns:
        rdmap_LR(tensor): low-resolution rdmaps, whose resolution has been reduced according to the reduction coefficient
    """
    # Auto-detect device if not provided
    if device is None:
        device = rdmaps.device
    rdmaps = rdmaps.to(device)

    assert len(rdmaps.shape) == 4
    
    number_of_range_bins, number_of_pulses = rdmaps.shape[1], rdmaps.shape[2]

    # bring rdmaps to time domain for Doppler and to frequency domain for Range
    rdmaps = undo_fft_RangeDoppler(rdmaps) if undo_fft else rdmaps
    
    if velocity: 
        lr_map_min_velocity_shape, lr_map_max_velocity_shape = get_middle_part_rdmap(number_of_pulses, k)
        rdmaps = rdmaps[:, :,  int(lr_map_min_velocity_shape):int(lr_map_max_velocity_shape), :]
    if range:
        # remove the high frequencies
        rdmaps = rdmaps[:, :number_of_range_bins//k, :, :]

    # back to the original domains
    rdmap_LR = fft_RangeDoppler(rdmaps) if undo_fft else rdmaps
    return rdmap_LR


def zeropadding_rdmap(lr_rdmaps, velocity=True, range=True, k=2, sr_range=128, sr_velocity=128, undo_fft=False, device=None):
    """
    Perform the Super Resolution inside the processing chain by adding zeros:
        -  at the beginning and end of the doppler profile in time domain
        -  at the end of the range spectrum (range profile in freq domain)
    The lr maps should be of shape:  (n_maps, n_range_cells, n_doppler_cells, n_angle_cells) 

    Args:
        lr_rdmaps       : the low-resolution LR Maps
        velocity (bool) : whether to reduce the resolution accross velocity axis. Defaults to True
        range (bool)    : whether to reduce the resolution accross range axis. Defaults to True
        k (int)         : The reduction coefficient of lowering the resolution. Defaults to 2
        sr_range        : 
        sr_velocity     : 
        undo_fft (bool) : whether to undo the fft on the range-doppler maps
        device          : torch device to run operations on (auto-detected if None)
    Returns:
        rdmap_SR(tensor): super-resolved rdmaps, whose resolution has been increased via zero padding
    """
    # Auto-detect device if not provided
    if device is None:
        device = lr_rdmaps.device
        
    assert len(lr_rdmaps.shape) == 4
    
    num_range_bins, num_doppler_bins = sr_range, sr_velocity

    # bring rdmaps to time domain for Doppler and to frequency domain for Range
    rdmaps = undo_fft_RangeDoppler(lr_rdmaps) if undo_fft else lr_rdmaps
    SR_range_doppler_map = torch.zeros((rdmaps.shape[0], num_range_bins, num_doppler_bins, rdmaps.shape[-1]), 
                                       dtype=torch.complex64, device=device)

    if range and velocity:
        lr_map_min_velocity_shape, lr_map_max_velocity_shape = get_middle_part_rdmap(num_doppler_bins, k)
        # Range and Velocity SR
        SR_range_doppler_map[:, :sr_range//k, int(lr_map_min_velocity_shape):int(lr_map_max_velocity_shape), :] = rdmaps
       
    if velocity and not range:
        lr_map_min_velocity_shape, lr_map_max_velocity_shape = get_middle_part_rdmap(num_doppler_bins, k)
        # Velocity resolution
        SR_range_doppler_map[:, :, int(lr_map_min_velocity_shape):int(lr_map_max_velocity_shape), :] = rdmaps

    if range and not velocity:
        # Range SR
        SR_range_doppler_map[:, :sr_range//k, :, :] = rdmaps

    # back to original domain
    rdmaps_SR = fft_RangeDoppler(SR_range_doppler_map) if undo_fft else SR_range_doppler_map
    return rdmaps_SR