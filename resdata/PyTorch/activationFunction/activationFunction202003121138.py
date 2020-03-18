# -*- coding: utf-8 -*-
"""
@Project : activationFunction
@Author  : Xu-Shan Zhao
@Filename: activationFunction202003121138.py
@IDE     : PyCharm
@Time1   : 2020-03-12 11:38:55
@Time2   : 2020/3/12 11:38
@Month1  : 3月
@Month2  : 三月
"""

import torch
import matplotlib.pyplot as plt

x_data = torch.arange(-6, 6, 0.01)
y_tanh = torch.tanh(x_data)
y_sigmoid = torch.sigmoid(x_data)
y_relu = torch.relu(x_data)
y_leakyrelu = torch.nn.functional.leaky_relu(x_data, negative_slope=0.05)
y_prelu = torch.prelu(x_data, weight=torch.tensor(0.25))
y_rrelu = torch.rrelu(x_data, lower=0., upper=1)

plt.ion()

plt.cla()
plt.plot(x_data.data.numpy(), y_tanh.data.numpy(), c='red', label='tanh')
plt.legend()
plt.xlabel('x')
plt.ylabel('tanh(x)')
plt.pause(1)

plt.cla()
plt.plot(x_data.data.numpy(), y_sigmoid.data.numpy(), c='red', label='sigmoid')
plt.legend()
plt.xlabel('x')
plt.ylabel('Sigmoid(x)')
plt.pause(1)

plt.cla()
plt.plot(x_data.data.numpy(), y_relu.data.numpy(), c='red', label='ReLU')
plt.legend()
plt.xlabel('x')
plt.ylabel('ReLU(x)')
plt.pause(1)

plt.cla()
plt.plot(x_data.data.numpy(), y_leakyrelu.data.numpy(), c='red', label='Leaky ReLU')
plt.legend()
plt.xlabel('x')
plt.ylabel('Leaky ReLU(x)')
plt.pause(1)

plt.cla()
plt.plot(x_data.data.numpy(), y_prelu.data.numpy(), c='red', label='PReLU')
plt.legend()
plt.xlabel('x')
plt.ylabel('PReLU(x)')
plt.pause(1)

plt.cla()
plt.plot(x_data.data.numpy(), y_rrelu.data.numpy(), c='red', label='RReLU')
plt.legend()
plt.xlabel('x')
plt.ylabel('RReLU(x)')
plt.pause(1)

plt.cla()
plt.plot(x_data.data.numpy(), y_tanh.data.numpy(), c='red', linestyle='-', label='tanh')
plt.plot(x_data.data.numpy(), y_sigmoid.data.numpy(), c='blue', linestyle='-.', label='sigmoid')
plt.plot(x_data.data.numpy(), y_relu.data.numpy(), c='green', linestyle='--', label='ReLU')
plt.plot(x_data.data.numpy(), y_leakyrelu.data.numpy(), c='black', linestyle=':', label='Leaky ReLU')
plt.plot(x_data.data.numpy(), y_prelu.data.numpy(), c='orange', label='PReLU')
plt.plot(x_data.data.numpy(), y_rrelu.data.numpy(), c='gold', label='RReLU')
plt.legend()
plt.ylim(-1.05, 1.5)
plt.xlabel('x')
plt.pause(1)

plt.ioff()
plt.show()
