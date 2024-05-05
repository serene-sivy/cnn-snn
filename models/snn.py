import torch
import torch.nn as nn
import torchvision
from spikingjelly.activation_based import neuron, functional, surrogate, layer

class CSNN(nn.Module):
    def __init__(self, T: int, channels: int, use_cupy=False):
        super().__init__()
        self.T = T

        self.conv_fc = nn.Sequential(
        layer.Conv2d(2, channels, kernel_size=3, padding=1, bias=False),
        layer.BatchNorm2d(channels),
        neuron.IFNode(surrogate_function=surrogate.Sigmoid()),
        layer.MaxPool2d(2, 2),  # 17 * 17

        layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        layer.BatchNorm2d(channels),
        neuron.IFNode(surrogate_function=surrogate.Sigmoid()),
        layer.MaxPool2d(2, 2),  # 8 * 8

        layer.Flatten(),
        layer.Linear(channels * 8 * 8, channels * 4 * 4, bias=False),
        neuron.IFNode(surrogate_function=surrogate.Sigmoid()),

        layer.Linear(channels * 4 * 4, 10, bias=False),
        neuron.IFNode(surrogate_function=surrogate.Sigmoid()),
        )

        functional.set_step_mode(self, step_mode='m')
        if use_cupy:
            functional.set_backend(self, backend='cupy')

    def forward(self, x: torch.Tensor):
        # x.shape = [N, C, H, W]
        # x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]
        # fr = 0
        # for t in range(self.T):
        #     x_seq = self.conv_fc(x[t])
        #     fr += x_seq
        x_seq = self.conv_fc(x)
        fr = x_seq.mean(0)
        return fr