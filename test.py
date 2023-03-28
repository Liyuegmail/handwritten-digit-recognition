import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image


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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

model = CNN().to(device)
model.load_state_dict(torch.load('model30_Adam.pth', map_location=torch.device('cpu')))

model.eval()

with torch.no_grad():
    image = Image.open('0.png')
    image = transform(image).unsqueeze(0).to(device)
    output = model(image)
    _, predicted = torch.max(output.data, 1)

    print(f'Predicted Digit: {predicted.item()}')
