# -*- coding: utf-8 -*-
"""
@Project : stockLSTM
@Author  : Xu-Shan Zhao
@Filename: stockPresLSTM202003161639.py
@IDE     : PyCharm
@Time1   : 2020-03-16 16:39:16
@Time2   : 2020/3/16 16:39
@Month1  : 3月
@Month2  : 三月
"""
import torch
from torch import nn, optim
import pandas as pd
import seaborn as sns
import numpy as np
import tushare as ts
import matplotlib.pyplot as plt

stock601600 = ts.get_hist_data('601600')
high = stock601600['high']
# high.plot(legend=True)
# plt.show()
high = (high - high.mean()) / (high.max() - high.min())
high = high.sort_index(ascending=True)

# Setting super parameters
learning_rate = 0.01
n_epoches = 300
h_state = None
input_size = 90

df = pd.DataFrame()
for i in range(input_size):
    df['c%d' % i] = high.tolist()[i: -input_size + i]

df.at[len(high) - input_size] = high.tolist()[-input_size:]

x = torch.FloatTensor(df.iloc[:, :].to_numpy())
x = x.unsqueeze(0)
y = high[input_size - 1:]
y = torch.FloatTensor(y.to_numpy())
y = y.unsqueeze(0).unsqueeze(2)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=90,
            num_layers=2,
        )
        self.out = nn.Linear(90, 1)

    def forward(self, x, h):
        out, h = self.rnn(x, h)
        prediction = self.out(out)
        return prediction, h


model = Net()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

plt.figure()
plt.ion()
plt.pause(2)

for epoch in range(n_epoches):
    prediction, h = model(x, h_state)
    optimizer.zero_grad()
    loss = criterion(prediction, y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(epoch + 1, loss.item())
        plt.cla()
        plt.plot(np.arange(x.shape[1]), y.view(-1).data.numpy(), 'ro')
        plt.plot(np.arange(x.shape[1]), prediction.view(-1).data.numpy())
        plt.draw()
        plt.pause(0.01)

plt.ioff()
plt.show()
