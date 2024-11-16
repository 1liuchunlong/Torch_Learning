import torch
import matplotlib.pyplot as plt
# 1. prepare dateset 
# 2. Design model using class(nn.Module)
# 3. Construct loss and optimizer
# 4. Train (forward backward ...)


# 注意 mini_batch 输入
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])

class LinearModel(torch.nn.Module):
    def __init__(self):
        # 调佣父类的__init__
        super(LinearModel,self).__init__()
        #  in_feature out_feature bias = True
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        # __call__ magic method 可调用的对象
        y_pred = self.linear(x)
        return y_pred

model = LinearModel()
criterion = torch.nn.MSELoss(size_average=False)
# 需要对linear的 w和 b 的参数的优化(linear.parameters())
# 申明 需要对那些参数优化 以及学习率设置
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

for epoch in range(100):
    #  都是callable的对象
    y_pred = model(x_data) # forward : predict
    loss = criterion(y_pred, y_data) # 注意 loss 是 标量
    print(epoch, loss.item())
    # 梯度 归零
    optimizer.zero_grad()
    # 反向传播
    loss.backward()
    # 权重更新
    optimizer.step()

print(f'w = {model.linear.weight.item()}')
print(f'b = {model.linear.bias.item()}')

# test Model
x_test = torch.Tensor([4.0])
print(model(x_test).data)