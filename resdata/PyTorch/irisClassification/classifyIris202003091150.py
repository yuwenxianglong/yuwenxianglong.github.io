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

iris = pd.read_csv('iris.data')
fts = iris.iloc[:, 0:4]
irk = iris.iloc[:, 4]

xfts = torch.FloatTensor(fts.to_numpy())
yirk = torch.LongTensor(irk.to_numpy())

epoches = 5000
learning_rate = 1e-2


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


model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epoches):
    optimizer.zero_grad()
    y_pred = model(xfts)
    loss = criterion(y_pred, yirk)
    loss.backward()
    optimizer.step()

    prediction = torch.max(y_pred, 1)[1]
    pred_y = prediction.data.numpy()
    target_y = yirk.data.numpy()
    accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)

    print(epoch, '\t', loss.item(), '\t', accuracy)
