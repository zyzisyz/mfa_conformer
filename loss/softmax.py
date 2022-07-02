import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .utils import accuracy

class softmax(nn.Module):
    def __init__(self, embedding_dim, num_classes, **kwargs):
        super(softmax, self).__init__()
        self.embedding_dim = embedding_dim
        self.fc = nn.Linear(embedding_dim, num_classes)
        self.criertion = nn.CrossEntropyLoss()

        print('init softmax')
        print('Embedding dim is {}, number of speakers is {}'.format(embedding_dim, num_classes))

    def forward(self, x, label=None):
        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.embedding_dim

        x = F.normalize(x, dim=1)
        x = self.fc(x)
        loss = self.criertion(x, label)
        acc1 = accuracy(x.detach(), label.detach(), topk=(1,))[0]
        return loss, acc1


if __name__ == "__main__":
    model = softmax(10, 100)
    data = torch.randn((2, 10))
    label = torch.tensor([0, 1])
    loss, acc = model(data, label)

    print(data.shape)
    print(loss)
    print(acc)

