# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# single layer
class SingleLayer(nn.Module):
    def __init__(self, nChannels: int, growthRate: int, use_dropout: bool):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels,
                               growthRate,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x: torch.tensor):
        out = self.conv1(F.relu(x, inplace=True))
        if self.use_dropout:
            out = self.dropout(out)
        out = torch.cat((x, out), 1)
        return out


# transition layer
class Transition(nn.Module):
    def __init__(self, nChannels: int, nOutChannels: int, use_dropout: bool):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nOutChannels)
        self.conv1 = nn.Conv2d(nChannels,
                               nOutChannels,
                               kernel_size=1,
                               bias=False)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x: torch.tensor):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        out = F.avg_pool2d(out, 2, ceil_mode=True)
        return out


# DenseNet-B
class Bottleneck(nn.Module):
    def __init__(self, nChannels: int, growthRate: int, use_dropout: bool):
        super(Bottleneck, self).__init__()
        interChannels = 4 * growthRate
        self.bn1 = nn.BatchNorm2d(interChannels)
        self.conv1 = nn.Conv2d(nChannels,
                               interChannels,
                               kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(growthRate)
        self.conv2 = nn.Conv2d(interChannels,
                               growthRate,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x: torch.tensor):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        out = torch.cat((x, out), 1)
        return out


class DenseNet(nn.Module):
    def __init__(self,
                 growthRate: int = 24,
                 reduction: float = 0.5,
                 bottleneck: bool = True,
                 use_dropout: bool = True):
        super(DenseNet, self).__init__()
        nDenseBlocks = 16
        nChannels = 2 * growthRate
        self.conv1 = nn.Conv2d(1,
                               nChannels,
                               kernel_size=7,
                               padding=3,
                               stride=2,
                               bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks,
                                       bottleneck, use_dropout)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans1 = Transition(nChannels, nOutChannels, use_dropout)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks,
                                       bottleneck, use_dropout)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans2 = Transition(nChannels, nOutChannels, use_dropout)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks,
                                       bottleneck, use_dropout)

    def _make_dense(self, nChannels: int, growthRate: int, nDenseBlocks: int,
                    bottleneck: bool, use_dropout: bool):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate, use_dropout))
            else:
                layers.append(SingleLayer(nChannels, growthRate, use_dropout))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x: torch.tensor):
        out = self.conv1(x)
        # out_mask = x_mask[:, 0::2, 0::2]
        out = F.relu(out, inplace=True)
        out = F.max_pool2d(out, 2, ceil_mode=True)
        # out_mask = out_mask[:, 0::2, 0::2]
        out = self.dense1(out)
        out = self.trans1(out)
        # out_mask = out_mask[:, 0::2, 0::2]
        out = self.dense2(out)
        out = self.trans2(out)
        # out_mask = out_mask[:, 0::2, 0::2]
        out = self.dense3(out)
        return out


if __name__ == "__main__":
    model = DenseNet(growthRate=24,
                     reduction=0.5,
                     bottleneck=True,
                     use_dropout=True)
    print(model)
    import time
    test_data = torch.rand(1, 1, 320, 320)
    starttime = time.time()
    for i in range(100):
        test_outputs = model(test_data)
        print(test_outputs.shape)
    endtime = time.time()
    print("Cost time ", endtime - starttime)

    import os
    SAVE_ONNX_MODEL_PATH = os.path.join(os.path.dirname(__file__),
                                        "../checkpoints")
    torch.onnx.export(model, test_data,
                      os.path.join(SAVE_ONNX_MODEL_PATH, "densnet.onnx"))