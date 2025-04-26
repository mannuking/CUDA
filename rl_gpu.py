import torch
import gymnasium as gym

# Device
assert torch.cuda.is_available(), "CUDA GPU is required for this test!"
device = torch.device('cuda')
print(f'Using device: {device}')

# RL: CartPole-v1 Random Policy (CPU for env, GPU for tensor ops)
env = gym.make('CartPole-v1')
obs, _ = env.reset()
obs = torch.tensor(obs, dtype=torch.float32).to(device)
for step in range(10):
    action = torch.randint(0, 2, (1,)).item()
    obs, reward, terminated, truncated, _ = env.step(action)
    obs = torch.tensor(obs, dtype=torch.float32).to(device)
    print(f"Step {step+1}, Action: {action}, Reward: {reward}")
    if terminated or truncated:
        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32).to(device)
print("\nRL test completed on GPU (for tensor ops)!") 
