import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import datasets
import torch.optim as optim
import torch.nn.functional as F

### 1. prepare datasets
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))]) # 归一化
train_dataset = datasets.MNIST(root = './dataset/mnist/', train = True, download = False, transform = transform)
test_dataset = datasets.MNIST(root = './dataset/mnist/', train = False, download = False, transform = transform)
train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True, num_workers = 0)
test_loader = DataLoader(test_dataset, batch_size = 64, shuffle = False, num_workers = 0) 


### 2. Design Model class


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)
model = Model()

### 3. construct Loss and Optimizer

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.5)


### 4. train cycle

def train(epoch):
    running_loss = 0
    for batch_idx, (inputs, target) in enumerate(train_loader, 0):
        optimizer.zero_grad()

        ## forward + backward + update
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print(f'epoch  {epoch + 1} batch_idx {batch_idx + 1} loss {running_loss / 300}')
            running_loss = 0.0

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for (inputs, labels) in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim = 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'ACC is {correct * 100 / total}')

if __name__ == '__main__':
    for epoch in range(100):
        train(epoch)
        test()