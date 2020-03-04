# -*- coding: utf-8 -*-
"""
@Project : combined_Cycle_Power_Plant
@Author  : Xu-Shan Zhao
@Filename: 202003021816LinearRegression.py
@IDE     : PyCharm
@Time1   : 2020-03-02 18:17:14
@Time2   : 2020/3/2 6:17 下午
@Month1  : 3月
@Month2  : 三月
"""

# 参考：http://cn.voidcc.com/question/p-eiuddbob-vb.html

import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt

ccpp = pd.read_csv('Folds5x2_pp.csv')

fts = ccpp.iloc[:, 0:4]
target = ccpp.iloc[:, 4]

xfts = torch.FloatTensor(fts.to_numpy())
ytarget = torch.FloatTensor(target.to_numpy())
ytarget = ytarget.unsqueeze(1)

input_size = xfts.shape[1]
output_size = 1
num_epochs = 3000
learning_rate = 2e-2

class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        return out

model = LinearRegression(input_size, output_size)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

plt.ion()

for epoch in range(num_epochs):
    inputs = xfts
    targets = ytarget
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    preds = model(xfts)

    print('epoch [%d/%d], Loss: %.4f'
          % (epoch + 1, num_epochs, loss.item()))
    plt.cla()
    plt.plot(ytarget.data.numpy(), preds.data.numpy(), 'r*',
             ytarget.data.numpy(), ytarget.data.numpy(), 'b-')
    # plt.axis('scaled')
    plt.axis('equal')
    plt.pause(0.0001)

plt.ioff()
plt.show()