import numpy as np
import matplotlib.pyplot as plt
import torch
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as sn
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm.std import trange
from torch.utils.data import dataloader


class DownScale(torch.nn.Module):
    def __init__(self):
        super(DownScale, self).__init__()
        self.layer1 = torch.nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1,padding=1)
        self.layer2 = torch.nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, stride=1,padding=1)
        self.layer3 = torch.nn.MaxPool2d(2)
        self.layer4 = torch.nn.Conv2d(in_channels=10, out_channels=1, kernel_size=3, stride=1,padding=1)
        self.layer5 = torch.nn.MaxPool2d(2)
        self.PReLU = torch.nn.PReLU()
    def forward(self, x):
        x = self.layer1(x)
        x = self.PReLU(x)
        x = self.layer2(x)
        x = self.PReLU(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.PReLU(x)
        x = self.layer5(x)
        x = self.PReLU(x)
        return x
