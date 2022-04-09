import numpy as np
import torch

def compute_dB(waveform):
    """
    Args:
        x (torch.tensor): Input waveform (#length).
    Returns:
        torch.tensor: Output array (#length).
    """
    val = max(0.0, torch.mean(torch.pow(waveform, 2)))
    dB = 10*torch.log10(val+1e-4)
    return dB

def compute_SNR(waveform, noise):
    """
    Args:
        x (numpy.array): Input waveform (#length).
    Returns:
        numpy.array: Output array (#length).
    """
    SNR = 10*np.log10(np.mean(waveform**2)/np.mean(noise**2)+1e-9)
    return SNR


