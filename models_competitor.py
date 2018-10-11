import torch.nn as nn
import torch.nn.functional as F
from typing import *


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1,  padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(32),
            nn.PReLU())

        self.conv_2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(64),
            nn.PReLU())

        self.conv_3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU())

        self.conv_4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU())

        self.conv_5 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU())

        self.conv_6 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(256),
            nn.PReLU())

        self.conv_7 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU())

        self.conv_8 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU())

        self.deconv_1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 6, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.PReLU())

        self.deconv_2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 6, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.PReLU())

        self.deconv_3 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, 6, stride=2, padding=2),
            nn.BatchNorm2d(1),
            nn.Tanh())


    def forward(self, x):

        # Encoder
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.conv_6(x)
        x = self.conv_7(x)
        x = self.conv_8(x)

        # Decoder
        x = self.deconv_1(x)
        x = self.deconv_2(x)
        x = self.deconv_3(x)

        return x

class Discriminator(nn.Module):
    def __init__(self, dataset_name, target_size):
        super(Discriminator, self).__init__()

        self.fc_dim = 512 * (target_size // 32) * (target_size // 32)
        self.dataset_name = dataset_name

        self.conv_d_1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1,  padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.conv_d_2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1,  padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU())

        self.conv_d_3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1,  padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())

        self.conv_d_4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1,  padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU())

        self.conv_d_5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=1,  padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())

        self.conv_d_6 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1,  padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU())

        self.conv_d_7 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1,  padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())

        self.conv_d_8 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1,  padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU())


        self.fc1 = nn.Linear(self.fc_dim, 2048)
        self.fc2 = nn.Linear(2048, 2048)

        if self.dataset_name == 'pandora':
            self.fc3 = nn.Linear(2048, 100)
        elif self.dataset_name == 'biwi':
            self.fc3 = nn.Linear(2048, 24)

    def forward(self, x):
        x = self.conv_d_1(x)
        x = self.conv_d_2(x)
        x = self.conv_d_3(x)
        x = self.conv_d_4(x)
        x = self.conv_d_5(x)
        x = self.conv_d_6(x)
        x = self.conv_d_7(x)
        x = self.conv_d_8(x)

        x = x.view(-1, self.fc_dim)

        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))

        return x
