import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from scipy.io import wavfile
from scipy import signal
import soundfile

def compute_dB(waveform):
    """
    Args:
        x (numpy.array): Input waveform (#length).
    Returns:
        numpy.array: Output array (#length).
    """
    val = max(0.0, np.mean(np.power(waveform, 2)))
    dB = 10*np.log10(val+1e-4)
    return dB

class WavAugment(object):
    def __init__(self, noise_csv_path="data/noise.csv", rir_csv_path="data/rir.csv"):
        self.noise_paths = pd.read_csv(noise_csv_path)["utt_paths"].values
        self.noise_names = pd.read_csv(noise_csv_path)["speaker_name"].values
        self.rir_paths = pd.read_csv(rir_csv_path)["utt_paths"].values

    def __call__(self, waveform):
        idx = np.random.randint(0, 10)
        if idx == 0:
            waveform = self.add_gaussian_noise(waveform)
            waveform = self.add_real_noise(waveform)

        if idx == 1 or idx == 2 or idx == 3:
            waveform = self.add_real_noise(waveform)

        if idx == 4 or idx == 5 or idx == 6:
            waveform = self.reverberate(waveform)

        if idx == 7:
            waveform = self.change_volum(waveform)
            waveform = self.reverberate(waveform)

        if idx == 6:
            waveform = self.change_volum(waveform)
            waveform = self.add_real_noise(waveform)

        if idx == 8:
            waveform = self.add_gaussian_noise(waveform)
            waveform = self.reverberate(waveform)

        return waveform

    def add_gaussian_noise(self, waveform):
        """
        Args:
            x (numpy.array): Input waveform array (#length).
        Returns:
            numpy.array: Output waveform array (#length).
        """
        snr = np.random.uniform(low=10, high=25)
        clean_dB = compute_dB(waveform)
        noise = np.random.randn(len(waveform))
        noise_dB = compute_dB(noise)
        noise = np.sqrt(10 ** ((clean_dB - noise_dB - snr) / 10)) * noise
        waveform = (waveform + noise)
        return waveform

    def change_volum(self, waveform):
        """
        Args:
            x (numpy.array): Input waveform array (#length).
        Returns:
            numpy.array: Output waveform array (#length).
        """
        volum = np.random.uniform(low=0.8, high=1.0005)
        waveform = waveform * volum
        return waveform

    def add_real_noise(self, waveform):
        """
        Args:
            x (numpy.array): Input length (#length).
        Returns:
            numpy.array: Output waveform array (#length).
        """
        clean_dB = compute_dB(waveform)

        idx = np.random.randint(0, len(self.noise_paths))
        sample_rate, noise = wavfile.read(self.noise_paths[idx])
        noise = noise.astype(np.float64)

        snr = np.random.uniform(15, 25)

        noise_length = len(noise)
        audio_length = len(waveform)

        if audio_length >= noise_length:
            shortage = audio_length - noise_length
            noise = np.pad(noise, (0, shortage), 'wrap')
        else:
            start = np.random.randint(0, (noise_length-audio_length))
            noise = noise[start:start+audio_length]

        noise_dB = compute_dB(noise)
        noise = np.sqrt(10 ** ((clean_dB - noise_dB - snr) / 10)) * noise
        waveform = (waveform + noise)
        return waveform

    def reverberate(self, waveform):
        """
        Args:
            x (numpy.array): Input length (#length).
        Returns:
            numpy.array: Output waveform array (#length).
        """
        audio_length = len(waveform)
        idx = np.random.randint(0, len(self.rir_paths))

        path = self.rir_paths[idx]
        rir, sample_rate = soundfile.read(path)
        rir = rir/np.sqrt(np.sum(rir**2))

        waveform = signal.convolve(waveform, rir, mode='full')
        return waveform[:audio_length]


if __name__ == "__main__":
    aug = WavAugment()
    sample_rate, waveform = wavfile.read("input.wav")
    waveform = waveform.astype(np.float64)

    gaussian_noise_wave = aug.add_gaussian_noise(waveform)
    print(gaussian_noise_wave.dtype)
    wavfile.write("gaussian_noise_wave.wav", 16000, gaussian_noise_wave.astype(np.int16))

    real_noise_wave = aug.add_real_noise(waveform)
    print(real_noise_wave.dtype)
    wavfile.write("real_noise_wave.wav", 16000, real_noise_wave.astype(np.int16))

    change_volum_wave = aug.change_volum(waveform)
    print(change_volum_wave.dtype)
    wavfile.write("change_volum_wave.wav", 16000, change_volum_wave.astype(np.int16))

    reverberate_wave = aug.reverberate(waveform)
    print(reverberate_wave.dtype)
    wavfile.write("reverberate_wave.wav", 16000, reverberate_wave.astype(np.int16))

    reverb_noise_wave = aug.reverberate(waveform)
    reverb_noise_wave = aug.add_real_noise(waveform)
    print(reverb_noise_wave.dtype)
    wavfile.write("reverb_noise_wave.wav", 16000, reverb_noise_wave.astype(np.int16))

    noise_reverb_wave = aug.add_real_noise(waveform)
    noise_reverb_wave = aug.reverberate(waveform)
    print(noise_reverb_wave.dtype)
    wavfile.write("noise_reverb_wave.wav", 16000, reverb_noise_wave.astype(np.int16))

    a = torch.FloatTensor(noise_reverb_wave)
    print(a.dtype)
