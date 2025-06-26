import idx2numpy
import numpy as np
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from cnn_utils import CNN, augment, normalize, training_loop, eval_loop

# Get the current working directory
current_directory = os.getcwd()

# Load MNIST dataset     
X_train = current_directory+'/mnist/train-images.idx3-ubyte'
y_train = current_directory+'/mnist/train-labels.idx1-ubyte'
X_test = current_directory+'/mnist/t10k-images.idx3-ubyte'
y_test = current_directory+'/mnist/t10k-labels.idx1-ubyte'

X_train = idx2numpy.convert_from_file(X_train)
y_train = idx2numpy.convert_from_file(y_train)
X_test = idx2numpy.convert_from_file(X_test)
y_test = idx2numpy.convert_from_file(y_test)


#Augment and normalize data
X_train, y_train = augment(X_train, y_train)
X_test, y_test = augment(X_test, y_test)

X_train, y_train = normalize(X_train, y_train)
X_test, y_test = normalize(X_test, y_test)

#Make into tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)


#Datasets and Loaders
train_dataset = TensorDataset(X_train_tensor.unsqueeze(1), y_train_tensor)
test_dataset = TensorDataset(X_test_tensor.unsqueeze(1), y_test_tensor)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

#Define model
model = CNN(1, 10)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

training_loop(model, optimizer=optimizer, num_epochs=num_epochs)