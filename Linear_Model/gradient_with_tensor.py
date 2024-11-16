import torch 
import matplotlib.pyplot as plt
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.tensor([1.0])
w.requires_grad = True

def forward(x):
    return x * w

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

loss_list = []
epoch_list = []

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()
        # 反向传播 计算梯度 存在 w.grad.data
        w.data = w.data - 0.01 * w.grad.data
        # 梯队 清零
        w.grad.data.zero_()
    epoch_list.append(epoch + 1)
    loss_list.append(l.item())

plt.figure(figsize=(10, 8))
plt.plot(epoch_list, loss_list, marker = 'o', linestyle = '-', linewidth = 2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('After training 100 epochs')
plt.grid(True)
plt.show()

print(f'Predict x = 4.0 y = {forward(4).item()}')