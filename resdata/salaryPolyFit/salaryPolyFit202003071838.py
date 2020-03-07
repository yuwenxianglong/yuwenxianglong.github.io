# -*- coding: utf-8 -*-
"""
@Project : salaryPolyFit
@Author  : Xu-Shan Zhao
@Filename: salaryPolyFit202003071838.py.py
@IDE     : PyCharm
@Time1   : 2020-03-07 18:38:27
@Time2   : 2020/3/7 6:38 下午
@Month1  : 3月
@Month2  : 三月
"""

import pandas as pd
import torch
from torch import nn, optim
import matplotlib.pyplot as plt

sly = pd.read_csv('Salary_Data.csv')
n = 3
learning_rate = 0.1
epoches = 3000000

yearsExperience = sly.iloc[:, 0]
salary = sly.iloc[:, 1]

x_yearsExperience = torch.FloatTensor(yearsExperience)
y_salary = torch.FloatTensor(salary)

x_yearsExperience = x_yearsExperience.unsqueeze(1)
y_salary = y_salary.unsqueeze(1)
x_yearsExperience = torch.cat([x_yearsExperience ** i
                               for i in range(1, n + 1)], 1)


class poly_model(nn.Module):
    def __init__(self):
        super(poly_model, self).__init__()
        self.poly = nn.Linear(n, 1)

    def forward(self, x):
        out = self.poly(x)
        return out


try:
    model = torch.load('salaryPR.pth')
except FileNotFoundError:
    model = poly_model()
except EOFError:
    model = poly_model()

criterion = nn.MSELoss()
optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

plt.ion()

for epoch in range(epoches):
    optimizer.zero_grad()
    outputs = model(x_yearsExperience)
    loss = criterion(outputs, y_salary)
    loss.backward()
    optimizer.step()
    preds = model(x_yearsExperience)
    if (epoch + 1) % 1000 == 0:
        print(epoch + 1, '\t', loss.item())

        plt.cla()
        plt.plot(yearsExperience, salary, 'ro', yearsExperience, preds.data.numpy(), 'b-')
        # plt.plot(y_salary.data.numpy(), preds.data.numpy(), 'ro',
        #          y_salary.data.numpy(), y_salary.data.numpy(), 'b-')
        plt.pause(0.1)

    if (epoch + 1) % 100 == 0:
        torch.save(model, './salaryPR.pth')

plt.ioff()
plt.show()
