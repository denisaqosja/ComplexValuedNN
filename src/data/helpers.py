import numpy as np
from numpy import linspace,pi,arange,arcsin,sin,square,zeros, tile
from scipy import ndimage, signal
    
    
def calculate_receiver_power(Pt, chirp_length, Gain, wavelength, Losses):
    """
    Calculate the received power using the corresponding formula 
    Return: 
        - r_power: power at the receiver
    """
    r_power = Pt * chirp_length * Gain**2 *wavelength**2 / ( (4*pi)**3  * Losses)   

    return r_power

def get_frequency_span(n_targets, fc, bandwidth, number_of_range_bins):
    """
    Calculate the span of frequencies, as a vector, between fc-B/2 and fc+B/2
    """
    freqVec = linspace(fc-bandwidth/2, fc+bandwidth/2, number_of_range_bins)   
    freqVec = np.tile(freqVec, (n_targets, 1))
    return freqVec


def get_delay_from_range(n_targets, range_r, c_0, fc, bandwidth, number_of_range_bins): 
    # get the frequency range and the delay
    frequencies = get_frequency_span(n_targets, fc, bandwidth, number_of_range_bins)
    delay = 2  * range_r / c_0

    return delay, frequencies


def get_ULA_elements_position(n_targets, number_ula_elements, ula_spacing):
    """ 
    Determine the position of the antenna elements
    Return:
        antenna_element_pos_x: the position of the elements in the x-range
    """
    # D: first define the index of the antenna elements and calculate their position in x-axis
    ula_array = arange(0.0, number_ula_elements) - number_ula_elements/2 + 0.5
    # determine the position (x-axis) of the ULA antenna elements
    antenna_element_pos_x = ula_array * ula_spacing
    # antenna_element_pos_x = np.reshape(antenna_element_pos_x, (1, 1, self.n_anntenna_elements))
    antenna_element_pos_x = tile(antenna_element_pos_x, (n_targets, 1))
    return antenna_element_pos_x


def get_angle_phase_shift(n_targets, theta, wavelength, number_ula_elements, ula_spacing):
    # calculate the angle phase shift: 2pi * f/c * ula_spacing * sin(theta) * N, where N is the number of antenna array elements
    # obtain the position of ULA elements in the x-axis
    antenna_element_pos_x = get_ULA_elements_position(n_targets, number_ula_elements, ula_spacing)
    f_angle = 2 * pi * sin(theta) * antenna_element_pos_x / wavelength
    
    return f_angle


def get_range_bin(bw, c_0, number_of_range_bins):
    """
    Change bandwidth given a new bandwidth value.
    Correspondingly change the wavenumber the range resolution, the range, and waveform number k
    Args:
        - bw: bandwidth
    Return:
        - the range bins, which are equidistant relative to the range resolution 
    """
    # calculate the range resolution; why isn't the ambiguity taken in consider? Maybe because the range bins are defined based on the fft points ... ?
    range_res = c_0 / (2 * bw)
    max_range = number_of_range_bins * range_res
    # max range: not used here, as the number of range bins is set by default to 256; r_max = c * PRI / 2
    range_bins_vec = arange(0, number_of_range_bins) * range_res

    return range_bins_vec, max_range


def get_velocity_bin(num_of_pulses, PRF, wavelength):
    """
    Obtain the axis of velocity for the Range-Doppler Map.  Calculate the following:
        - maximum velocity
            fd_max = PRF / 2 ; v_max = lambda * fd_max / 2 = lambda * PRF / 4 
            HERE: v_max only divided by 2
        - velocity resolution 
            v_res = v_max / 
    
    Args:
        - num of pulses: the new number of pulses to transmit/receive
    """
    # determine the velocity resolution
    v_max = PRF * wavelength / 4
    # determine v_res; v_max/v_res = n_pulses/2  ; here n_pulses-1 instead of n_pulses to keep one more resolution bin 
    v_res = 2 * v_max / (num_of_pulses - 1)
    # define the velocity bins, from -v_max : v_max
    vel_bins_vec = arange(0, num_of_pulses) * v_res - v_max

    return vel_bins_vec


def get_angle_bin(n, wavelength, ula_spacing):
    """
    Calculate angular resolution and angles bin array (similar to the range bins)
    Args: 
        - n: 
    Return:
        - angle_bins_vec: a vector/array with the bins of the angle which depends on the angular resolution
    """
    # the number of FFT points are usually set to => N anntena elements in ULA
    nfft_angle = n
    # angular resolution 
    angle_res = wavelength/ (n * ula_spacing)
    # angle values: but not used anywhere for building the map
    angle_bins_vec = -arcsin(angle_res*(arange(0,n)-0.5*n))/pi*180

    return angle_bins_vec



def get_doppler_frequency(n_targets, velocity, number_of_pulses, wavelength, PRF):
    """
    Obtain the doppler shift
    """
    # create a sequence of number of pulses
    pulse_sequence = linspace(0.0, number_of_pulses, number_of_pulses)
    pulse_sequence_targets = tile(pulse_sequence, (n_targets, 1))
    # removing the dependency of fd from time; and converting it into a dependency of number of pulses. Derived at the notebook ...
    fd = 2 * velocity * pulse_sequence_targets / (wavelength * PRF)

    return fd


def calculate_SNR_after_LR(profile, noise, Fs, SNR, reduction_coeff, periodogram=False):
    """Calculate the new SNR of the LR profile after removing certain pulses
    Args:
        profile: the original high-resolution profile
        profile_lr: the new profile with lower resolution 
        noise: awgn noise
        reduction_coeff: the coefficient by which the number of pulses is reduced, e.g. 2
        periodogram: bool value whether to use periodogram or not
    """
    # Use the periodogram to calculate the power spectral density (PSD) of the signal, and use the PSD to calculate the SNR
    if periodogram:
        # perform the periodogram to obtain the power spectral density 
        f, pxx = signal.periodogram(profile, Fs)
        # calculate the new PSD given the reduction coeff, e.g can be half of the original one
        pxx_new_LR = pxx * 1/reduction_coeff
        # calculate the noise power (remains unchanged) and use it in the SNR relying on the Parseval's theorem
        noise_power = square(np.abs(noise))
        SNR_new = pxx_new_LR / noise_power
    else:
        SNR_new = SNR * 1 / reduction_coeff
    
    return SNR_new


def normalize_range_bin_vector(profile, range_bins_vec, number_of_range_bins):
    """Normalize the HRR profile wrt the range

    Args:
        profile: A HRR profile with shape(n_range_bins, n_vel_bins, n_angle_bins)
        range_bins_vec: the range bins in a vector format 
        number_of_range_bins: the number of range bins 

    Returns:
        profile: A normalized HRR profile in terms of range, so the dependency of range 
    """
    # the power of signal attenuates with a factor of 1/range^4 as it travels away from the radar
    normalized_range_bins_vec = np.power(range_bins_vec, 4)
    normalized_range_bins_vec = np.reshape(normalized_range_bins_vec, (number_of_range_bins, 1))
    # all targets are brought to similar powers regardless of the range;  
    profile = profile * normalized_range_bins_vec  # element-wise multiplication
    
    return profile