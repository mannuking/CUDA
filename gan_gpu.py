import torch
import torch.nn as nn
import torch.optim as optim

# Device
assert torch.cuda.is_available(), "CUDA GPU is required for this test!"
device = torch.device('cuda')
print(f'Using device: {device}')

# Hyperparameters
z_dim = 16
hidden_dim = 32
batch_size = 128
num_epochs = 5

# Simple synthetic data: 1D Gaussian
real_data = torch.randn(batch_size, 1).to(device)

# Generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, z):
        return self.net(z)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

gen = Generator().to(device)
disc = Discriminator().to(device)
loss_fn = nn.BCELoss()
gen_opt = optim.Adam(gen.parameters(), lr=0.001)
disc_opt = optim.Adam(disc.parameters(), lr=0.001)

for epoch in range(num_epochs):
    # Train Discriminator
    z = torch.randn(batch_size, z_dim).to(device)
    fake_data = gen(z)
    real_labels = torch.ones(batch_size, 1).to(device)
    fake_labels = torch.zeros(batch_size, 1).to(device)
    disc_real = disc(real_data)
    disc_fake = disc(fake_data.detach())
    loss_real = loss_fn(disc_real, real_labels)
    loss_fake = loss_fn(disc_fake, fake_labels)
    disc_loss = (loss_real + loss_fake) / 2
    disc_opt.zero_grad()
    disc_loss.backward()
    disc_opt.step()
    # Train Generator
    z = torch.randn(batch_size, z_dim).to(device)
    fake_data = gen(z)
    disc_fake = disc(fake_data)
    gen_loss = loss_fn(disc_fake, real_labels)
    gen_opt.zero_grad()
    gen_loss.backward()
    gen_opt.step()
    print(f"Epoch {epoch+1}, D Loss: {disc_loss.item():.4f}, G Loss: {gen_loss.item():.4f}") 
