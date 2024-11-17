import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import datasets
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

### 1. prepare datasets
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))]) # 归一化
train_dataset = datasets.MNIST(root = './dataset/mnist/', train = True, download = False, transform = transform)
test_dataset = datasets.MNIST(root = './dataset/mnist/', train = False, download = False, transform = transform)
train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True, num_workers = 0)
test_loader = DataLoader(test_dataset, batch_size = 64, shuffle = False, num_workers = 0) 

### 2. design model

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size = 5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size = 5)
        self.pooling = torch.nn.MaxPool2d(kernel_size = 2)
        self.fc = torch.nn.Linear(320, 10)
    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1)
        return  self.fc(x)

model = Net()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

### 3. construct criterion and optimizer

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum= 0.5)


### 4. train cycle forward backward update

def train(epoch):
    running_loss = 0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print(f'epoch = {epoch + 1}  batch_idx = {batch_idx + 1}  loss = {running_loss / 300}')
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
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    print(f'ACC is {100 * correct / total} %')
    return correct / total

if __name__ == '__main__':
    epoch_list = []
    acc_list = []
    for epoch in range(10):
        train(epoch)
        epoch_list.append(epoch + 1)
        acc_list.append(test())
    plt.plot(epoch_list, acc_list)
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.show()