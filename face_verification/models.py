from __future__ import division, print_function
import torch as t
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sigmoid


def freeze_layer(layer):
    for param in layer.parameters():
        param.requires_grad = False


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, batch_norm=False,
                 activation_fn=nn.ReLU, dropout_prob=0.):
        super().__init__()

        self.net = nn.Sequential()

        self.net.add_module('Conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                              kernel_size=kernel_size, stride=stride, padding=padding,
                                              dilation=dilation))

        if batch_norm:
            self.net.add_module('Batch Norm', nn.BatchNorm2d(out_channels))

        self.net.add_module('Activation Function', activation_fn())

        if dropout_prob > 0:
            self.net.add_module('Dropout', nn.Dropout2d(p=dropout_prob))

    def forward(self, x):
        return self.net(x)


class FCBlock(nn.Module):
    def __init__(self, in_features, out_features, bias=True, batch_norm=False, activation_fn=nn.ReLU, dropout_prob=0.):
        super().__init__()

        self.net = nn.Sequential()

        self.net.add_module('FC', nn.Linear(in_features=in_features, out_features=out_features, bias=bias))

        if batch_norm:
            self.net.add_module('Batch Norm', nn.BatchNorm2d(out_features))

        self.net.add_module('Activation Function', activation_fn())

        if dropout_prob > 0:
            self.net.add_module('Dropout', nn.Dropout(p=dropout_prob))

    def forward(self, x):
        return self.net(x)


class ResidualConnection(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, batch_norm=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.batch_norm = batch_norm

        self.net = nn.Sequential()

        if stride != 1 or self.in_channels != self.out_channels:
            self.net.add_module(
                'Residual_Conv2d',
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=stride, bias=False)
            )
            if batch_norm:
                self.net.add_module('Residual_Batch_Norm', nn.BatchNorm2d(self.out_channels))
        else:
            pass

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, conv_blocks, batch_norm=False):
        super().__init__()
        self.conv_blocks = conv_blocks
        self.batch_norm = batch_norm

        self.net = nn.Sequential()
        self.res_stride = 1
        for i, block in enumerate(self.conv_blocks):
            self.net.add_module('ResidualBlock_Conv' + str(i), block)
            self.res_stride *= block.net[0].stride[0]

        self.residual = \
            ResidualConnection(in_channels=self.conv_blocks[0].net[0].in_channels,
                               out_channels=self.conv_blocks[-1].net[0].out_channels,
                               stride=self.res_stride,
                               batch_norm=self.batch_norm)

    def forward(self, x):
        return self.net(x) + self.residual(x)


class FirstNet(nn.Module):
    def __init__(self, dropout_prob=0., layer_id=None):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.layer_id = layer_id

        self.conv1 = ConvBlock(in_channels=1, out_channels=64, kernel_size=(3, 3), batch_norm=True,
                               activation_fn=nn.ReLU, stride=2)
        # self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv2 = ConvBlock(in_channels=64, out_channels=128, kernel_size=(3, 3), batch_norm=True,
                               activation_fn=nn.ReLU, stride=2)
        # self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv3 = ConvBlock(in_channels=128, out_channels=256, kernel_size=(3, 3), batch_norm=True,
                               activation_fn=nn.ReLU, stride=2)
        self.conv4 = ConvBlock(in_channels=256, out_channels=256, kernel_size=(3, 3), batch_norm=True,
                               activation_fn=nn.ReLU, stride=2)
        # self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv5 = ConvBlock(in_channels=256, out_channels=256, kernel_size=(3, 3), batch_norm=True,
                               activation_fn=nn.ReLU, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc = FCBlock(in_features=512, out_features=128, activation_fn=nn.ReLU, dropout_prob=self.dropout_prob)
        self.fc1 = FCBlock(in_features=128, out_features=32, activation_fn=nn.ReLU, dropout_prob=self.dropout_prob)
        self.fc2 = FCBlock(in_features=32, out_features=1, activation_fn=nn.Sigmoid)

        if self.layer_id:
            self.freezed_layers = [layer for i, layer in enumerate([
                self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.avg_pool, self.fc, self.fc1, self.fc2,
            ]) if i < self.layer_id]

    def freeze_layers(self):
        for layer in self.freezed_layers:
            freeze_layer(layer)

    def branch(self, x):
        x = self.conv1(x)
        if self.layer_id == 1:
            ret = x
        # x = self.pool1(x)

        x = self.conv2(x)
        if self.layer_id == 2:
            ret = x
        # x = self.pool2(x)

        x = self.conv3(x)
        if self.layer_id == 3:
            ret = x
        x = self.conv4(x)
        if self.layer_id == 4:
            ret = x
        # x = self.pool4(x)

        x = self.conv5(x)
        if self.layer_id == 5:
            ret = x

        # x = F.adaptive_avg_pool2d(x, (1, 1))
        x = self.avg_pool(x)

        if self.layer_id in (1, 2, 3, 4, 5):
            return x, ret
        else:
            return x, None

    def forward(self, x1, x2):
        y1, conv1 = self.branch(x1)
        y2, conv2 = self.branch(x2)

        y1 = t.squeeze(y1)
        y2 = t.squeeze(y2)

        y = t.cat((y1, y2), dim=-1)

        y = self.fc(y)
        fc = y

        y = self.fc1(y)
        y = self.fc2(y)

        if self.layer_id in (1, 2, 3, 4, 5):
            return y, conv1, conv2, fc
        else:
            return y


class JanusNet(nn.Module):
    def __init__(self, network_model, dropout_prob=0.25, layer_id=5):
        super().__init__()

        self.network_model = network_model
        self.dropout_prob = dropout_prob
        self.layer_id = layer_id

        self.depth_net = self.network_model(dropout_prob=dropout_prob, layer_id=self.layer_id)
        self.hybrid_net = self.network_model(dropout_prob=dropout_prob, layer_id=self.layer_id)
        self.rgb_net = self.network_model(dropout_prob=dropout_prob, layer_id=self.layer_id)

    def forward(self, d1, d2, rgb1, rgb2):
        y_depth, _, _, _ = self.depth_net(d1, d2)
        y_hallucination, conv_a_hallucination, conv_b_hallucination, _ = self.hybrid_net(d1, d2)
        y_rgb, conv_a_rgb, conv_b_rgb, _ = self.rgb_net(rgb1, rgb2)

        if self.training:
            return y_depth, y_hallucination, y_rgb, conv_a_hallucination, conv_b_hallucination, conv_a_rgb, conv_b_rgb
        else:
            return y_depth, y_hallucination, y_rgb
