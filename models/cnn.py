import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.conv_fc = nn.Sequential(
        nn.Conv2d(2, channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(channels),
        nn.Sigmoid(),
        nn.MaxPool2d(2, 2),  # 17 * 17

        nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(channels),
        nn.Sigmoid(),
        nn.MaxPool2d(2, 2),  # 8 * 8

        nn.Flatten(),
        nn.Linear(channels * 8 * 8, channels * 4 * 4, bias=False),
        nn.Sigmoid(),

        nn.Linear(channels * 4 * 4, 10, bias=False),
        nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        # x.shape = [N, C, H, W]
        x = x.repeat(1, 2, 1, 1) # to align with snn input channel
        out = self.conv_fc(x)
        return out