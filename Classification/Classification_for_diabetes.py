import torch 
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

### 1. prepare datasets
data = np.loadtxt('./dataset/diabetes/diabetes.csv', delimiter=',', dtype= np.float32)
x_data = torch.from_numpy(data[:, : -1])
y_data = torch.from_numpy(data[:, [-1]])
### 2. design model

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 2)
        self.linear4 = torch.nn.Linear(2, 1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        x = self.sigmoid(self.linear4(x))
        return x
model = Model()

### 3. construct criterion and optimizer 
criterion = torch.nn.BCELoss(size_average = True)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

### 4. train cycle

for epoch in range(1000000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    # print(f'epoch = {epoch + 1}  loss = {loss.item()}')
    # backward
    optimizer.zero_grad()
    loss.backward()
    # update
    optimizer.step()
    if epoch % 100000 == 99999:
        y_pred_label = torch.where(y_pred >= 0.5, torch.Tensor([1.0]), torch.Tensor([0.0]))
        acc = torch.eq(y_pred_label, y_data).sum() / y_data.size(0)
        print(f'loss = {loss.item()}, acc = {acc}')