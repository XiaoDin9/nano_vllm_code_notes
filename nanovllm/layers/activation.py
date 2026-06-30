import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 假设 x 的形状为 (batch_size, n)，则分割后：
        # x 的形状为 (batch_size, n//2), y 的形状为 (batch_size, n - n//2)（如果 n 是奇数，y 将会比 x 多一个元素）。
        x, y = x.chunk(2, -1)

        # F.silu(x) 计算 SiLU 激活函数的输出, 公式: x / (1 + e^(-x))
        return F.silu(x) * y
