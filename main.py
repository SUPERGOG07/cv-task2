import time

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

import resnet

epochs = 100
batch_size_train = 128
batch_size_test = 100
learning_rate = 1e-3
momentum = 0.9
weight_decay = 1e-3
mean = [0.5070746, 0.48654896, 0.44091788]
std = [0.26733422, 0.25643846, 0.27615058]

train_transform = transforms.Compose([

    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(45),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

train_dataset = torchvision.datasets.CIFAR100('./data/cifar100/', train=True, download=True, transform=train_transform)
test_dataset = torchvision.datasets.CIFAR100('./data/cifar100/', train=False, download=True, transform=test_transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size_train, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size_test, shuffle=True)

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

network = resnet.resnet18().to(device)

print(network)

# 损失函数
loss_func = nn.CrossEntropyLoss().to(device)
# 优化器
optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
# 学习率调度
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


def train(epoch):
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
        correct += predicted.eq(labels).sum()

        if i % 200 == 199:
            print('[%d, %5d] loss: %.3f accuracy: %.2f %%' % (epoch, i + 1, loss.item(), 100 * correct / total))
            # torch.save(network.state_dict(), './network/model.pth')
            # torch.save(optimizer.state_dict(), './network/optimizer.pth')

    print('Training completed.')


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
            correct += predicted.eq(labels).sum()

    print('Testing completed.')
    print('Accuracy on test set: %.2f %%' % (100 * correct / total))


if __name__ == '__main__':
    for epoch in range(1, epochs + 1):
        start_time = time.time()

        print('=' * 30)
        train(epoch)
        test()

        end_time = time.time()
        run_time = end_time - start_time
        print('time used: %.2f minutes' % (run_time / 60))
