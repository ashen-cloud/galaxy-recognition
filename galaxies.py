#!/usr/bin/python

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def to_categorical(y, num_classes=None, dtype='float32'):
  y = np.array(y, dtype='int')
  input_shape = y.shape
  if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
    input_shape = tuple(input_shape[:-1])
  y = y.ravel()
  if not num_classes:
    num_classes = np.max(y) + 1
  n = y.shape[0]
  categorical = np.zeros((n, num_classes), dtype=dtype)
  categorical[np.arange(n), y] = 1
  output_shape = input_shape + (num_classes,)
  categorical = np.reshape(categorical, output_shape)
  return categorical

with h5py.File('Galaxy10_DECals.h5', 'r') as File:
    images = np.array(File['images'])
    labels = np.array(File['ans'])

labels = labels.astype(np.float64)
images = images.astype(np.float64)

images_f = np.einsum('klij->kjil', images)
labels_f = to_categorical(labels, 10)

split_idx = int(images.shape[0] * 0.8)

images = images_f[:split_idx, :]
images_test = images_f[split_idx:, :]

labels = labels_f[:split_idx, :]
labels_test = labels_f[split_idx:, :]


class GalaxyNet(nn.Module):

    def __init__(self):
        super(GalaxyNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 24, 5)
        self.maxpool = nn.MaxPool2d(kernel_size=3, padding=0)

        # self.conv2 = nn.Conv2d(18, 24, 5)
        self.conv3 = nn.Conv2d(24, 48, 5)
        self.conv4 = nn.Conv2d(48, 92, 5)
        # self.avgpool = nn.MaxPool2d(kernel_size=3, padding=0) # nn.AdaptiveAvgPool2d(output_size=1)

        self.fc1 = nn.Linear(92 * 26 * 26, 10)
        self.softmax1 = nn.LogSoftmax(dim=1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(F.leaky_relu(x))

        # x = self.conv2(x)
        # x = self.maxpool(F.leaky_relu(x))

        x = self.conv3(x)
        x = self.maxpool(F.leaky_relu(x))

        x = self.conv4(x)

        pooled = self.maxpool(F.leaky_relu(x))

        # pooled = pooled.view(x.shape[0], 92 * 26 * 26)

        # if not hasattr(self, 'fc1'):
        #     self.fc1 = nn.Linear(128 * 26 * 26, 10)

        x = self.fc1(pooled.view(pooled.size(0), -1))
        x = self.softmax1(x)
        return x


model = GalaxyNet()
# model.to('cuda:0')

optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 1
BATCH = 4

loss_fn = nn.CrossEntropyLoss()

for i in tqdm(range(EPOCHS)):
    optimizer.zero_grad()

    samp = np.random.randint(0, images.shape[0], (BATCH))

    X = torch.tensor(images[samp]).float()
    # X = torch.tensor(images[samp]).to('cuda:0').float()

    out = model(X)

    Y = torch.argmax(torch.tensor(labels[samp]).long(), dim=1)
    # Y = torch.argmax(torch.tensor(labels[samp]).long().to('cuda:0'), dim=1)
    
    loss = loss_fn(out, Y)

    loss.backward()

    optimizer.step()


total = 0
correct = 0
with torch.no_grad():
    for i in tqdm(range(images_test.shape[0])):
        X_test = images_test[i]
        Y_test = labels_test[i]

        X = torch.tensor(X_test).float()
        # X = torch.tensor(X_test).to('cuda:0').float()

        X = X.view(1, 3, 256, 256)

        out = model(X)

        prediction = torch.argmax(out)
        real = torch.argmax(torch.tensor(Y_test))
        # real = torch.argmax(torch.tensor(Y_test).to('cuda:0'))

        if prediction == real:
            correct += 1

        total += 1

print('total', total)
print('correct', correct)
print('accuracy', round(correct / total, 3))


