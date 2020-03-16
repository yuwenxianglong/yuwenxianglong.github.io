---
title: PyTorch之RNN拟合航班载客量
author: 赵旭山
tags: PyTorch
typora-root-url: ..
---





```python
data_csv = pd.read_csv('international-airline-passengers.csv', usecols=[1])
data_psger = data_csv.iloc[:, 0]
data_psger = (data_psger - data_psger.mean()) / (data_psger.max() - data_psger.min())

input_size = 3

df = pd.DataFrame()
for i in range(input_size):
    df['c%d' % i] = data_psger.tolist()[i: -input_size + i]

df.at[len(data_psger) - input_size] = data_psger.tolist()[-input_size:]

x = torch.FloatTensor(df.iloc[:, :].to_numpy())
x = x.unsqueeze(0)
y = data_psger[input_size - 1:]
y = torch.FloatTensor(y.to_numpy())
y = y.unsqueeze(0).unsqueeze(2)

n_epoches = 30000
learning_rate = 0.0001

epoches = []
pres = []


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
        out = out.view(s * b, h)
        prediction = self.out(out)
        prediction = prediction.view(s, b, -1)
        return prediction, h


model = Net()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_func = nn.MSELoss()
h_state = None

plt.figure()
plt.ion()
plt.pause(5)

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