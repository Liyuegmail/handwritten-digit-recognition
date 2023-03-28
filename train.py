import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

learning_rate = 0.001
batch_size = 64
num_epochs = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.model = nn.Sequential(  # 放在序列当中
            nn.Conv2d(1, 32, 3, 1, 1),  # 输入通道，输出通道,核大小3，走1步，填充1
            nn.MaxPool2d(2),  # 变成原来的一半
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.Flatten(),  # 展平
            nn.Linear(64 * 7 * 7, 1024),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model = CNN().to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()  # 梯度清零

        outputs = model(images)  # 前向传播

        loss = loss_fn(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        running_loss += loss.item()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}] , Step [{}/{}],loss:{:.4f}'.format(epoch + 1, num_epochs, i + 1, len(train_loader),
                                                                    running_loss / 100))
            running_loss = 0.0

    # 没有梯度的时候评估模型
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print('Epoch [{}/{}] Test Accuracy:{:.4f}'.format(epoch + 1, num_epochs, accuracy))
    torch.save(model.state_dict(), 'model{}.pth'.format(epoch))
