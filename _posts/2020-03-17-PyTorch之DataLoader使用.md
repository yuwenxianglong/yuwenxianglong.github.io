---
title: PyTorch之Dataloader使用
author: 赵旭山
tags: PyTorch
typora-root-url: ..
---

目前看，只有在处理图像分类例子中有实际意义，对时序数据无实际意义，暂不展开了。

![](/assets/images/dataloaderDescription202003171941.png)

参考文章中的示例也先照搬过来吧：

```python
"""
    批训练，把数据变成一小批一小批数据进行训练。
    DataLoader就是用来包装所使用的数据，每次抛出一批数据
"""
import torch
import torch.utils.data as Data

BATCH_SIZE = 5

x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)
# 把数据放在数据库中
torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    # 从数据库中每次抽出batch size个样本
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
)


def show_batch():
    for epoch in range(3):
        for step, (batch_x, batch_y) in enumerate(loader):
            # training


            print("steop:{}, batch_x:{}, batch_y:{}".format(step, batch_x, batch_y))


if __name__ == '__main__':
    show_batch()
```







#### References：

* [torch.utils.data.DataLoader使用方法](https://www.cnblogs.com/demo-deng/p/10623334.html)
