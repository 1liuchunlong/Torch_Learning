import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import datasets
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
### 1. prepare datasets
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))]) # 归一化
train_dataset = datasets.MNIST(root = './dataset/mnist/', train = True, download = False, transform = transform)
test_dataset = datasets.MNIST(root = './dataset/mnist/', train = False, download = False, transform = transform)
train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True, num_workers = 0)
test_loader = DataLoader(test_dataset, batch_size = 64, shuffle = False, num_workers = 0) 

### 2. desigin model class
class InceptionA(nn.Module):
    def __init__(self, in_channles):
        super(InceptionA, self).__init__()
        self.batch_pool = nn.Conv2d(in_channels=in_channles, out_channels= 24, kernel_size= 1)

        self.batch1x1 = nn.Conv2d(in_channels=in_channles, out_channels= 16, kernel_size= 1)
        
        self.batch5x5_1 = nn.Conv2d(in_channels=in_channles, out_channels= 16, kernel_size= 1)
        self.batch5x5_2 = nn.Conv2d(16, 24, kernel_size= 5, padding= 2)

        self.batch3x3_1 = nn.Conv2d(in_channels= in_channles, out_channels= 16, kernel_size= 1)
        self.batch3x3_2 = nn.Conv2d(in_channels= 16, out_channels= 24, kernel_size= 3, padding= 1)
        self.batch3x3_3 = nn.Conv2d(in_channels= 24, out_channels= 24, kernel_size= 3, padding= 1)
    def forward(self, x):
        batch_pool = F.avg_pool2d(input= x, kernel_size= 3, stride= 1, padding= 1)
        batch_pool = self.batch_pool(batch_pool)

        batch1x1 = self.batch1x1(x)

        batch5x5 = self.batch5x5_1(x)
        batch5x5 = self.batch5x5_2(batch5x5)

        batch3x3 = self.batch3x3_1(x)
        batch3x3 = self.batch3x3_2(batch3x3)
        batch3x3 = self.batch3x3_3(batch3x3)

        outputs = [batch_pool, batch1x1, batch3x3, batch5x5]
        return torch.cat(outputs, dim= 1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size= 5)
        self.conv2 = nn.Conv2d(88, 20, kernel_size= 5)
        #  输出都是 88
        self.incep1 = InceptionA(in_channles= 10)
        self.incep2 = InceptionA(in_channles= 20)

        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(1408, 10)
    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = self.incep1(x)
        x= F.relu(self.mp(self.conv2(x)))
        x = self.incep2(x)
        x = x.view(in_size, -1)
        return self.fc(x)

model = Net()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

### 3. construct criterion and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum= 0.5)

### 4. train cycle  forward backward and update

def train(epoch):
    running_loss = 0
    for bacth_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if bacth_idx % 300 == 299:
            print(f'epoch = {epoch + 1} batch_idx = {bacth_idx + 1} loss = {running_loss / 300}')
            running_loss = 0

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim = 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'ACC is {100 * correct / total }%')
    return correct / total


if __name__ == '__main__':
    acc_list = []
    epoch_list = []
    for epoch in range(10):
        train(epoch)
        acc = test()
        epoch_list.append(epoch + 1)
        acc_list.append(acc)

    plt.figure(figsize==(10,8))
    plt.plot(epoch_list, acc_list)
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.title('Train cucle of Acc')
    plt.grid()
    plt.show()
