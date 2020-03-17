---
title: PyTorch之RNN拟合时序数据
author: 赵旭山
tags: PyTorch
typora-root-url: ..
---



所从事专业中大量涉及时序数据，单纯的线性拟合无法满足数据分析的要求。琢磨了两天，看了十数篇文章和代码，才算是对循环神经网络（RNN）有了粗浅的理解，

以下为一个具有代表性的时序`csv`格式数据，第一列为时间列表（月份），第二列为国际航班旅客数量（单位为“千 人”，in units of 1,000）。数据包括了从1949年1月到1960年12月共计12年144条数据。

#### 1. 读入数据

```python
data_csv = pd.read_csv('international-airline-passengers.csv', usecols=[1])
data_psger = data_csv.iloc[:, 0]
```

#### 2. 数据归一化

本文中所用到的数据集必须对数据进行归一化，否则训练过程无法开展。

> 《**[特征工程中的「归一化」有什么作用？](https://www.zhihu.com/question/20455227)**》：
>
> * 如果对**输出结果范围有要求**，用归一化；
>
> * 如果**数据较为稳定，不存在极端的最大和最小值**，用归一化；
> * 如果**数据存在异常值和较多噪音**，用标准化，可以间接通过中心化避免异常值和极端值的影响

本数据集归一化处理：

```
data_psger = (data_psger - data_psger.mean()) / (data_psger.max() - data_psger.min())
```

#### 3. 定义时序宽度

通俗理解，**新数据受之前多少个数据的影响**，有点像**周期**的概念。

```python
input_size = 3
df = pd.DataFrame()
for i in range(input_size):  # 定义时序宽度
    df['c%d' % i] = data_psger.tolist()[i: -input_size + i]

df.at[len(data_psger) - input_size] = data_psger.tolist()[-input_size:]  # 上述代码生成的数据序列会丢掉最后一个数据，所以要加上
```

#### 4. 转为PyTorch张量

```python
x = torch.FloatTensor(df.iloc[:, :].to_numpy())
x = x.unsqueeze(0)
y = data_psger[input_size - 1:]
y = torch.FloatTensor(y.to_numpy())
y = y.unsqueeze(0).unsqueeze(2)
```

#### 5. 构造网络模型

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=2,
            num_layers=1,
        )
        self.out = nn.Linear(2, 1)

    def forward(self, x, h):
        out, h = self.rnn(x, h)
        s, b, h = out.size()
        out = out.view(s * b, h)  # 没什么用，把out转为二维数据，以输入Linear，可以删掉此行
        prediction = self.out(out)
        prediction = prediction.view(s, b, -1)  # 没什么用，把结果prediction转回三维
        return prediction, h
```

#### 6. 模型训练与可视化

```python
n_epoches = 30000
learning_rate = 0.0001

epoches = []
pres = []

model = Net()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_func = nn.MSELoss()
h_state = None

plt.figure()
plt.ion()
plt.pause(2)

for epoch in range(n_epoches):
    prediction, h = model(x, h_state)
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(epoch, loss.item())
        plt.cla()
        plt.plot(torch.arange(df.shape[0]).numpy(), y.view(-1).data.numpy(), 'ro')
        plt.plot(torch.arange(df.shape[0]).numpy(), prediction.view(-1).data.numpy(), 'b-')
        plt.draw()
        plt.pause(0.1)

plt.ioff()
plt.show()
```



![](/assets/images/airlineRegressor202003162204.gif)



#### References：

* [Pytorch学习之LSTM预测航班](https://www.jianshu.com/p/18f397d908be)

* [归一化 （Normalization）、标准化 （Standardization）和中心化/零均值化 （Zero-centered）](https://www.jianshu.com/p/95a8f035c86c)
* [特征工程中的「归一化」有什么作用？](https://www.zhihu.com/question/20455227)