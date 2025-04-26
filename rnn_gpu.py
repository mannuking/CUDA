import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Device
assert torch.cuda.is_available(), "CUDA GPU is required for this test!"
device = torch.device('cuda')
print(f'Using device: {device}')

# Generate dummy sequential data (sequence classification)
N, T, D, C = 256, 10, 8, 2  # batch, time, features, classes
X = np.random.randn(N, T, D).astype(np.float32)
y = np.random.randint(0, C, size=(N,))
X_train = torch.tensor(X, dtype=torch.float32).to(device)
y_train = torch.tensor(y, dtype=torch.long).to(device)

# Model
class SimpleRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(D, 32, batch_first=True)
        self.fc = nn.Linear(32, C)
    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return self.fc(out)

model = SimpleRNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training
for epoch in range(5):
    optimizer.zero_grad()
    out = model(X_train)
    loss = loss_fn(out, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}") 
