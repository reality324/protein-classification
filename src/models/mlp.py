"""
MultiTask MLP 模型定义
共享编码器 + 多任务分类头 (EC + 定位 + 功能)
"""
import torch
import torch.nn as nn


class MultiTaskMLP(nn.Module):
    """多任务 MLP (共享编码器 + 多个任务头)"""
    
    def __init__(
        self,
        input_dim: int,
        output_dims: dict,  # {'ec': 75, 'loc': 7, 'func': 7}
        hidden_dims: list = [512, 256],
        dropout: float = 0.3
    ):
        super().__init__()
        
        # 共享编码器
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)
        
        # 任务特定头
        self.heads = nn.ModuleDict({
            task: nn.Linear(prev_dim, out_dim)
            for task, out_dim in output_dims.items()
        })
    
    def forward(self, x):
        h = self.encoder(x)
        return {
            task: head(h)
            for task, head in self.heads.items()
        }
