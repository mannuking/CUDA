import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

# Device
assert torch.cuda.is_available(), "CUDA GPU is required for this test!"
device = torch.device('cuda')
print(f'Using device: {device}')

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)

# Model
class SimpleCVNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16*15*15, 10)
        )
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

model = SimpleCVNet().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(2):
    optimizer.zero_grad()
    out = model(images)
    loss = loss_fn(out, labels)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Test (on the same batch for demo)
with torch.no_grad():
    pred = model(images).argmax(dim=1)
    acc = (pred == labels).float().mean().item()
    print(f"Batch Accuracy: {acc*100:.2f}%") 
