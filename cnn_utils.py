#IMPORTS THUS FAR

import idx2numpy
import numpy as np
import os
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


#THE MODEL

class CNN(nn.Module):
    def __init__(self, in_channels, classes=10):
        super(CNN, self).__init__()

        #first convolutional layer generating 8 feature maps
        self.conv1 = nn.Conv2d(in_channels= in_channels, out_channels=8, kernel_size=3, stride=1, padding=1)

        #max pooling layer to reduce the feature maps values by half
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #second convolutional layer generating 16 feature maps
        self.conv2 = nn.Conv2d(in_channels=8, out_channels = 16, kernel_size=3, stride=1, padding=1)

        #activation function
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.5)

        #linear layer to obtain the output class
        self.fc = nn.Linear(16*7*7, classes)

        #flattening layer to adjust tensor for linear layer
        self.flatten = nn.Flatten()

    def forward(self, x):
        first_conv = self.conv1(x)
        first_activation = self.relu(first_conv)
        first_pool = self.pool(first_activation)
        second_conv = self.conv2(first_pool)
        second_activation = self.relu(second_conv)
        second_pool = self.pool(second_activation)
        dropped = self.dropout(second_pool)
        flat = self.flatten(dropped)
        out = self.fc(flat)
        return out

def augment(x, y):
    flipped = np.flip(x, axis=2)
    rotated1 = np.rot90(x, k=1, axes=(1, 2))
    rotated2 = np.rot90(x, k=2, axes=(1, 2))
    rotated3 = np.rot90(x, k=3, axes=(1, 2))
    x = np.concatenate([x, flipped, rotated1, rotated2, rotated3], axis=0)
    y = np.concatenate([y, y, y, y, y], axis=0)
    return x, y