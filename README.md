# CUDA Speedtest & GPU ML/DL Test Suite

This project helps you verify your CUDA-enabled GPU setup and benchmark your system using various Machine Learning (ML), Deep Learning (DL), Reinforcement Learning (RL), and Computer Vision (CV) algorithms on the GPU.

## Features
- Check CUDA and GPU availability
- Compare CPU vs GPU speed for matrix multiplication
- Run sample ML/DL/AI/RL/CNN/DNN/RNN/GAN operations on the GPU
- Separate scripts for each major AI/ML/DL/RL task

## Requirements
- Python 3.8+
- NVIDIA GPU with CUDA support
- CUDA-enabled PyTorch
- See `requirements.txt` for all dependencies

## Installation
1. (Optional but recommended) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   For CUDA-enabled PyTorch, see [PyTorch Get Started](https://pytorch.org/get-started/locally/).

## Usage
Run any of the following scripts to test your GPU with different AI/ML/DL/RL workloads:
```bash
python cuda_speedtest.py         # Basic CUDA and speed test
python dnn_gpu.py               # Feedforward Neural Network (DNN)
python cnn_gpu.py               # Convolutional Neural Network (CNN)
python rnn_gpu.py               # Recurrent Neural Network (RNN)
python gan_gpu.py               # Generative Adversarial Network (GAN)
python rl_gpu.py                # Reinforcement Learning (CartPole, tensor ops on GPU)
python ml_gpu.py                # Classical ML (RandomForest, CPU only, with note)
python cv_gpu.py                # Computer Vision (CIFAR10 classification on GPU)
```

## Files
- `cuda_speedtest.py`: Basic CUDA and speed test script
- `dnn_gpu.py`: Feedforward Neural Network (DNN) on GPU
- `cnn_gpu.py`: Convolutional Neural Network (CNN) on GPU
- `rnn_gpu.py`: Recurrent Neural Network (RNN) on GPU
- `gan_gpu.py`: Generative Adversarial Network (GAN) on GPU
- `rl_gpu.py`: Reinforcement Learning (CartPole, tensor ops on GPU)
- `ml_gpu.py`: Classical ML (RandomForest, CPU only, with note)
- `cv_gpu.py`: Computer Vision (CIFAR10 classification on GPU)
- `requirements.txt`: List of required packages

## Notes
- Ensure you have the correct CUDA drivers and a compatible PyTorch version.
- All ML/DL tests are set to use the GPU if available (except `ml_gpu.py`, which uses CPU).

## License
MIT 
