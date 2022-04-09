import torch
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

class Model(torch.nn.Module):
    def __init__(self, n_mels=80, embedding_dim=192, channel=512):
        super(Model, self).__init__()
        channels = [channel for _ in range(4)]
        channels.append(channel*3)
        self.model = ECAPA_TDNN(input_size=n_mels, lin_neurons=embedding_dim, channels=channels)

    def forward(self, x):
        x = x.squeeze(1)
        x = x.permute(0, 2, 1)
        x = self.model(x)
        x = x.squeeze(1)
        return x
 
def ecapa_tdnn(n_mels=80, embedding_dim=192, channel=512):
    model = Model(n_mels=n_mels, embedding_dim=embedding_dim, channel=channel)
    return model

def ecapa_tdnn_large(n_mels=80, embedding_dim=192, channel=1024):
    model = Model(n_mels=n_mels, embedding_dim=embedding_dim, channel=channel)
    return model


