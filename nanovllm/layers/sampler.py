import torch
from torch import nn

'''
                logits
                  │
                  ▼
        ┌───────────────────┐
        │ divide temperature│
        └───────────────────┘
                  │
                  ▼
        ┌───────────────────┐
        │     softmax       │
        └───────────────────┘
                  │
                  ▼
        ┌───────────────────┐
        │ generate Exp(1)   │
        │ random matrix     │
        └───────────────────┘
                  │
                  ▼
        probs / random
                  │
                  ▼
        ┌───────────────────┐
        │      argmax       │
        └───────────────────┘
                  │
                  ▼
            next token
'''
# 采样：根据模型输出的 logits 按温度采样生成下一个 token
class Sampler(nn.Module):

    '''
    输入：
        logits: 模型输出，shape 通常是 [batch_size, vocab_size]
        temperatures: 每个样本的温度参数，shape [batch_size]

    输出：
        sample_tokens: 每个 batch 采样出的 token id，shape [batch_size]
    等价的实现：sample_tokens = torch.multinomial(probs, num_samples=1)
    '''
    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # temperatures.unsqueeze(dim=1)， eg: [0.8, 1.5] => [[0.8], [1.5]]
        # 广播除法，即:logits[0] 所有元素都 / 0.8, logits[1] 所有元素都 / 1.5
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        
        # softmax 转概率， 。它将一个向量转换为一个概率分布，输出的每个元素都在 0 到 1 之间，并且所有输出的和为 1
        probs = torch.softmax(logits, dim=-1)

        # torch.empty_like(probs).exponential_(1) 生成 Exp(1) 指数分布随机数
        # .clamp_min_(1e-10)， 防止除 0
        # probs / exponential_random
        # argmax 得到 token 0
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
