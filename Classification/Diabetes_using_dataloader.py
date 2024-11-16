import torch 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np


### 1. prepare datasets
class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        data = np.loadtxt(filepath, delimiter=',', dtype = np.float32)
        self.x_data = torch.from_numpy(data[:, : -1])
        self.y_data = torch.from_numpy(data[:, [-1]])
        self.len = data.shape[0]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        return self.len

dataset = DiabetesDataset('./dataset/diabetes/diabetes.csv')
train_loader = DataLoader(dataset = dataset, batch_size = 32, shuffle = True, num_workers = 2)

### Design model

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

criterion = torch.nn.BCELoss(reduction = 'mean') 
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

### 4. train cycle
if __name__ == '__main__':
    for epoch in range(100):
        for iteration, data in enumerate(train_loader, 0):
            inputs, labels = data
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(f'epoch = {epoch + 1}  iteration = {iteration}  loss = {loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  