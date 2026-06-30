import torch
from torch import nn


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    '''
    对输入向量 `x = [x_1, ..., x_D]`：
    rms(x) = sqrt((1 / D) * sum_{i=1}^{D} x_i^2 + eps)
    y_i = x_i / rms(x) * gamma_i
    其中 `gamma` 是可学习 scale，`eps` 是防止除零的小常数。当前图中常见 `eps=9.98377799987793e-07`
    '''
    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))                 # rsqrt(z),即 1 / sqrt(z), x.mul_(k),即 x1 = x1 * k

        # 用 b.mul_(a) 将会对向量 b 的每个元素与向量 a 的对应元素进行逐元素相乘。这个操作是原地进行的，会直接修改 b 的值
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())        # X[x1, x2, ..., xn] = X[x1, x2, ..., xn] + residual[r1, r2, ..., rn]
        residual = x.to(orig_dtype)

        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)

        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
