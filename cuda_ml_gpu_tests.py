import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import gymnasium as gym

# Set device to GPU if available
assert torch.cuda.is_available(), "CUDA GPU is required for these tests!"
device = torch.device('cuda')
print(f'Using device: {device}')

# 1. DNN: Simple Feedforward Neural Network on GPU
print("\n--- DNN: Feedforward Neural Network ---")
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)

class SimpleDNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.layers(x)

dnn = SimpleDNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(dnn.parameters(), lr=0.01)
for epoch in range(5):
    optimizer.zero_grad()
    out = dnn(X_train)
    loss = loss_fn(out, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 2. CNN: Simple Convolutional Neural Network on GPU
print("\n--- CNN: Simple CNN on MNIST ---")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8*13*13, 10)
        )
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

cnn = SimpleCNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.001)
dataiter = iter(trainloader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)
for epoch in range(2):
    optimizer.zero_grad()
    out = cnn(images)
    loss = loss_fn(out, labels)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 3. RL: Simple Gym Environment with Random Policy on GPU (if possible)
print("\n--- RL: CartPole-v1 Random Policy (CPU for env, GPU for tensor ops) ---")
env = gym.make('CartPole-v1')
obs, _ = env.reset()
obs = torch.tensor(obs, dtype=torch.float32).to(device)
for step in range(5):
    action = torch.randint(0, 2, (1,)).item()
    obs, reward, terminated, truncated, _ = env.step(action)
    obs = torch.tensor(obs, dtype=torch.float32).to(device)
    print(f"Step {step+1}, Action: {action}, Reward: {reward}")
    if terminated or truncated:
        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32).to(device)

print("\nAll tests completed on GPU!") 
