"""SB3 / PyTorch 计算设备（训练与推理共用，优先 CUDA）。"""
import torch

SB3_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
