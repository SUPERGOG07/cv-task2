import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

import resnet

epochs = 50
batch_size_train = 64
batch_size_test = 1000
learning_rate = 1e-3
momentum = 0.9

# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     # transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

# transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# ])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = torchvision.datasets.CIFAR100('./data/cifar100/', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR100('./data/cifar100/', train=False, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size_train, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size_test, shuffle=True)

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

network = resnet.resnet18().to(device)

print(network)

# 损失函数和优化器
loss_func = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)


def train(epoch):
    start_time = time.time()
    print('Training start')
    network.train()
    correct = 0
    total = 0
    for i, data in enumerate(train_loader, 0):
        images = data[0].to(device)
        labels = data[1].to(device)

        outputs = network(images)
        loss = loss_func(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if i % 200 == 199:
            print('[%d, %5d] loss: %.3f accuracy: %.2f %%' % (epoch, i + 1, loss.item(), 100 * correct / total))
            # torch.save(network.state_dict(), './network/model.pth')
            # torch.save(optimizer.state_dict(), './network/optimizer.pth')

    print('Training completed.')
    end_time = time.time()
    run_time = end_time - start_time
    print('time used: %.2f minutes' % (run_time / 60))


def test():
    print('Training start')
    network.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images = data[0].to(device)
            labels = data[1].to(device)
            outputs = network(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Testing completed.')
    print('Accuracy on test set: %.2f %%' % (100 * correct / total))


if __name__ == '__main__':
    for epoch in range(1, epochs + 1):
        print('=' * 30)
        train(epoch)
        test()
        print('=' * 30)
