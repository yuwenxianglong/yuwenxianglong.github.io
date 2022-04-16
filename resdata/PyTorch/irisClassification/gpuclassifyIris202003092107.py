# -*- coding: utf-8 -*-
"""
@Project : classifyIris
@Author  : Xu-Shan Zhao
@Filename: classifyIris202003091150.py
@IDE     : PyCharm
@Time1   : 2020-03-09 11:50:44
@Time2   : 2020/3/9 11:50 上午
@Month1  : 3月
@Month2  : 三月
"""

import torch
import pandas as pd
from torch import nn, optim
import matplotlib.pyplot as plt

iris = pd.read_csv('iris.data')
fts = iris.iloc[:, 0:4]
irk = iris.iloc[:, 4]

xfts = torch.FloatTensor(fts.to_numpy())
yirk = torch.LongTensor(irk.to_numpy())

num_epoches = 10000
learning_rate = 5e-3


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


try:
    model = torch.load('irisclfy.pth')
except FileNotFoundError:
    model = Net()
except EOFError:
    model = Net()

if torch.cuda.is_available():
    model = Net().cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

losses = []
epoches = []

plt.ion()
for epoch in range(num_epoches):
    if torch.cuda.is_available():
        xfts = xfts.cuda()
        yirk = yirk.cuda()

    optimizer.zero_grad()
    y_pred = model(xfts)
    loss = criterion(y_pred, yirk)
    loss.backward()
    optimizer.step()

    prediction = torch.max(y_pred, 1)[1]

    if torch.cuda.is_available():
        prediction = prediction.cpu()
        yirk = yirk.cpu()

    pred_y = prediction.data.numpy()
    target_y = yirk.data.numpy()
    accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)

    print(epoch, '\t', loss.item(), '\t', accuracy)

    if (epoch + 1) % 100 == 0:
        torch.save(model, 'irisclfy.pth')

    if (epoch + 1) % 10 == 0:
        losses.append(loss.item())
        epoches.append(epoch)
        plt.cla()
        plt.plot(epoches, losses)
        plt.pause(0.01)

plt.ioff()
plt.show()
