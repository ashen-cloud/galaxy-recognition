#!/usr/bin/python

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import cv2
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

labels = labels.astype(np.float32)
images = images.astype(np.float32)

# labels = labels[10000:]
# images = images[10000:]

# images_f = np.einsum('klij->kjil', images)
labels_f = to_categorical(labels, 10)

# resize = lambda x: cv2.resize(x, dsize=(60, 60), interpolation=cv2.INTER_CUBIC)
# images_f = np.array(map(resize, images_f))

images_res = np.empty((images.shape[0], 60, 60, 3), dtype=images.dtype)

for i in range(images.shape[0]):
    images_res[i] = cv2.resize(images[i], dsize=(60, 60), interpolation=cv2.INTER_CUBIC)

split_idx = int(images.shape[0] * 0.8)

images = images_res[:split_idx, :]
images_test = images_res[split_idx:, :]

labels = labels_f[:split_idx, :]
labels_test = labels_f[split_idx:, :]

# print('image shape', images[0].shape)
# quit()
class GalaxyNet(nn.Module):

    def __init__(self):
        super(GalaxyNet, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(3, 64, 9, stride=1, padding=int(5/2), padding_mode='zeros'),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 96, 5, stride=1, padding=int(5/2), padding_mode='zeros'),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.lins = nn.Sequential(
            nn.Linear(18816, 9408),
            nn.Sigmoid(),
            nn.Linear(9408, 4704),
            nn.Sigmoid(),
        )
        
        self.classifier = nn.Linear(4704, 10)
        
        self.dropout = nn.Dropout(0.15)


    def forward(self, x):
        x = self.convs(x)
        
        x = self.dropout(x)

        # print('shape', x.shape)
        x = x.reshape(x.size(0), -1)
        x = self.lins(x)
        # print('aft')

        x = self.classifier(x)

        return x


model = GalaxyNet()
model.to('cuda:0')

optimizer = optim.Adam(model.parameters(), lr=3e-4) # 3e-4 0.0003

EPOCHS = 30
BATCH = 24

loss_fn = nn.CrossEntropyLoss()

losses = []
accuracies = []

for i in tqdm(range(EPOCHS)):
    optimizer.zero_grad()

    samp = np.random.randint(0, images.shape[0], (BATCH))

    X = torch.tensor(images[samp]).to('cuda:0').float() / 255

    out = model(X.reshape(BATCH, 3, 60, 60))
    # print('actual', out)
    # print('probabilities', F.softmax(out, dim=1))

    Y = torch.tensor(labels[samp]).float().to('cuda:0')
    
    label = torch.argmax(Y, dim=1)

    cat = torch.argmax(out, dim=1)

    loss = loss_fn(out, label)

    accuracy = (cat == label).float().mean()

    losses.append(loss)
    accuracies.append(accuracy)

    loss.backward()

    optimizer.step()

for i in range(len(losses)):
    print('loss', losses[i])
    print('acc', accuracies[i])

total = 0
correct = 0
with torch.no_grad():
    test_data = range(images_test.shape[0])
    for i in tqdm(test_data):
        X_test = images_test[i]
        Y_test = labels_test[i]

        # X = torch.tensor(X_test).float()
        X = torch.tensor(X_test).to('cuda:0').float() / 255

        X = X.reshape(1, 3, 60, 60)

        out = model(X)

        prediction = torch.argmax(out)

        real = torch.argmax(torch.tensor(Y_test).to('cuda:0'))

        if prediction == real:
            correct += 1

        total += 1

print('total', total)
print('correct', correct)
print('accuracy', round(correct / total, 3))




# ------------------ JUNK ------------------ #
'''
self.conv1 = nn.Conv2d(3, 24, 5)
        self.maxpool = nn.MaxPool2d(kernel_size=3, padding=0)

        self.conv2 = nn.Conv2d(18, 24, 5)

        self.conv3 = nn.Conv2d(24, 48, 5)

        self.conv4 = nn.Conv2d(48, 92, 5)

        self.fc1 = nn.Linear(92 * 26 * 26, 10)
        self.softmax1 = nn.LogSoftmax(dim=1)


x = self.conv1(x)
        x = self.maxpool(F.leaky_relu(x))

        # x = self.conv2(x)
        # x = self.maxpool(F.leaky_relu(x))

        x = self.conv3(x)
        x = self.maxpool(F.leaky_relu(x))

        x = self.conv4(x)

        final = self.maxpool(F.leaky_relu(x))
        print('final shape', final.shape)

        # final = final.view(x.shape[0], 92 * 26 * 26)

        # if not hasattr(self, 'fc1'):
        #     self.fc1 = nn.Linear(128 * 26 * 26, 10)

        x = self.fc1(final.view(final.size(0), -1))
        x = self.softmax1(x)

'''


