# -*- coding: utf-8 -*-
from turtle import forward
import numpy
import torch
import torch.nn as nn


class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn: int = 256, nHidden: int = 256, nOut: int = 256):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

        self.nOut = nOut

    def forward(self, input: torch.tensor):
        recurrent, _ = self.rnn(input)
        T, B, hidden = recurrent.size()
        t_rec = recurrent.view(T * B, hidden)
        output = self.embedding(t_rec)
        output = output.view(T, B, -1)
        return output


class CNN(nn.Module):
    def __init__(self, enc_out_dim: int = 256):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(),
                                 nn.MaxPool2d(2, 2, 1),
                                 nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
                                 nn.MaxPool2d(2, 2, 1),
                                 nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(),
                                 nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(),
                                 nn.MaxPool2d((2, 1), (2, 1), 0),
                                 nn.Conv2d(128, enc_out_dim, 3, 1, 0),
                                 nn.ReLU())

        self.rnn = BidirectionalLSTM(256, 256, 256)

    def forward(self, imgs: torch.tensor):
        encoded_imgs = self.cnn(imgs)  # [B, 256, H', W']
        B, C, H, W = encoded_imgs.shape
        encoded_imgs = encoded_imgs.permute(3, 0, 2, 1)  # [W, B', H', 256]
        encoded_imgs = encoded_imgs.contiguous().view(W, B * H, -1)
        output = self.rnn(encoded_imgs)  # [W, B * H, 256]
        output = output.permute(1, 0, 2)  # [B * H, W, 256]
        encoded_imgs = output.contiguous().view(B, -1, H, W)
        return encoded_imgs


if __name__ == "__main__":
    model = CNN()
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
                      os.path.join(SAVE_ONNX_MODEL_PATH, "cnn.onnx"))
