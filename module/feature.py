import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F

class PreEmphasis(torch.nn.Module):
    def __init__(self, coef: float = 0.97):
        super(PreEmphasis, self).__init__()
        self.coef = coef
        # make kernel
        # In pytorch, the convolution operation uses cross-correlation. So, filter is flipped.
        self.register_buffer(
            'flipped_filter', torch.FloatTensor(
                [-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, inputs: torch.tensor) -> torch.tensor:
        assert len(
            inputs.size()) == 2, 'The number of dimensions of inputs tensor must be 2!'
        # reflect padding to match lengths of in/out
        inputs = inputs.unsqueeze(1)
        inputs = F.pad(inputs, (1, 0), 'reflect')
        return F.conv1d(inputs, self.flipped_filter).squeeze(1)


class Mel_Spectrogram(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=512, win_length=400, hop=160, n_mels=80, coef=0.97, requires_grad=False):
        super(Mel_Spectrogram, self).__init__()
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.win_length = win_length
        self.hop = hop

        self.pre_emphasis = PreEmphasis(coef)
        mel_basis = librosa.filters.mel(
            sr=sample_rate, n_fft=n_fft, n_mels=n_mels)
        self.mel_basis = nn.Parameter(
            torch.FloatTensor(mel_basis), requires_grad=requires_grad)
        self.instance_norm = nn.InstanceNorm1d(num_features=n_mels)
        window = torch.hamming_window(self.win_length)
        self.window = nn.Parameter(
            torch.FloatTensor(window), requires_grad=False)

    def forward(self, x):
        x = self.pre_emphasis(x)
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop,
                       window=self.window, win_length=self.win_length, return_complex=True)
        x = torch.abs(x)
        x += 1e-9
        x = torch.log(x)
        x = torch.matmul(self.mel_basis, x)
        x = self.instance_norm(x)
        x = x.unsqueeze(1)
        return x


if __name__ == "__main__":
    from scipy.io import wavfile
    import matplotlib.pyplot as plt
    from torchvision import transforms as transforms

    sample_rate, sig = wavfile.read("test.wav")
    sig = torch.FloatTensor(sig.copy())
    sig = sig.repeat(10, 1)

    spec = Mel_Spectrogram()
    out = spec(sig)
    out = out
    print(out.shape)

    plt.subplot(211)
    plt.imshow(out[0][0])

    trans = transforms.RandomResizedCrop((80, 200))
    out = trans(out)
    print(out.shape)

    plt.subplot(212)
    plt.imshow(out[0][0])

    plt.savefig("test.png")
