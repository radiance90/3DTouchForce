import torchvision
import torch
import numpy as np
import resnet_small2


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.feature = resnet_small2.resnet18_small(num_classes=128)
        self.feature.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)

        self.movement = torch.nn.Sequential(torch.nn.BatchNorm1d(256),
                                          torch.nn.Linear(256, 64),
                                          torch.nn.ReLU(),
                                          torch.nn.BatchNorm1d(64),
                                          torch.nn.Linear(64, 3))


    def forward(self, x):
        x1 = self.feature(x[:, 0, :, :][:, np.newaxis, :, :])
        x2 = self.feature(x[:, 1, :, :][:, np.newaxis, :, :])
        x_out = torch.cat([x1, x2], -1)
        moved = self.movement(x_out)
        return moved[:, 0], moved[:, 1], moved[:, 2]
