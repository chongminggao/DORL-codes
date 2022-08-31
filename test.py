# -*- coding: utf-8 -*-
# @Time    : 2022/7/19 19:12
# @Author  : Chongming GAO
# @FileName: test.py

import torch
from torch import nn


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.w1 = nn.Parameter(torch.tensor([2.], requires_grad=True))
        self.w2 = nn.Parameter(torch.tensor([3.], requires_grad=True))
        self.w3 = nn.Parameter(torch.tensor([4.], requires_grad=True))

    def forward(self, a):
        s = a * self.w1

        c = s * self.w2
        d = s * self.w3

        return c, d


x = torch.tensor([1.])
y1 = torch.tensor([10.])
y2 = torch.tensor([12.])

model = Model()
mse = torch.nn.MSELoss()
# optim = torch.optim.Adam(model.parameters(), lr=0.5)
optim = torch.optim.SGD(model.parameters(), lr=0.01)

num = 0
torch.autograd.set_detect_anomaly(True)  # For debugging.
while True:
    num += 1
    print(num)
    y1_pred, y2_pred = model(x)
    loss1 = mse(y1_pred, y1)

    print(y1_pred, y2_pred)

    optim.zero_grad()
    loss1.backward(retain_graph=True)
    optim.step()

    y1_pred, y2_pred = model(x)
    loss2 = mse(y2_pred, y2)

    optim.zero_grad()
    loss2.backward(retain_graph=True) # 报错行！while循环第二次运行到这里才会报错。为什么第一次不报错？
    optim.step()

    print(y1_pred, y2_pred)
    print(model(x))
    a = 1