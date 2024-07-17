import torch.nn as nn

"""
contains code to build models
"""

def ConvBlockBuilder(in_channels: int, out_channels: int, kernel_size: int | tuple, is_last: bool = False, padding = True):
    if not is_last:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=int(padding)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_channels),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=int(padding))
        )
    

def ConvBlock(in_channels: int, out_channels: int, kernel_size: int | tuple, is_last: bool = False, padding = True):
    return nn.Sequential(
        ConvBlockBuilder(in_channels, out_channels, kernel_size, padding=padding),
        ConvBlockBuilder(out_channels, out_channels, kernel_size, padding=padding),
        ConvBlockBuilder(out_channels, out_channels, kernel_size, is_last, padding)
    )


def createAudioModel():
    # i had to change the audio model bc dimensions :(
    return nn.Sequential(
        ConvBlock(1, 32, 2),
        nn.MaxPool2d(2, (4, 2)),
        ConvBlock(32, 64, 3),
        nn.MaxPool2d(2, (4, 2)),
        ConvBlock(64, 64, 3),
        nn.MaxPool2d(2, (4, 1)),
        ConvBlock(64, 64, 3),
        nn.MaxPool2d(2, (2, 1)),
        ConvBlock(64, 128, 3, is_last=True),
    )


def createVisualModel():
    return nn.Sequential(
        ConvBlock(5, 32, 3, padding=False),
        nn.MaxPool2d(2, (2, 2)),
        ConvBlock(32, 64, 3, padding=False),
        nn.MaxPool2d(2, (2, 2)),
        ConvBlock(64, 64, 3, padding=False),
        nn.MaxPool2d(2, (2, 2)),
        ConvBlock(64, 128, 3, is_last=True, padding=False),
    )


def createFusionModel():
    # simple concatenation for now, not done here
    return nn.Sequential()


def createFCModel():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(1024, 512),
        nn.LeakyReLU(),
        nn.Linear(512, 128),
        nn.LeakyReLU(),
        nn.Linear(128, 2)
    )