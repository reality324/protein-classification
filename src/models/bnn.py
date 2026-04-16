"""
Bayesian Neural Network 模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BayesianLinear(nn.Module):
    """贝叶斯全连接层 (使用 Flipout 或简化的 Dropout 近似)"""
    
    def __init__(self, in_features, out_features, dropout=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = nn.Dropout(dropout)
        
        # 均值和log方差
        self.weight_mean = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_log_std = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias_mean = nn.Parameter(torch.zeros(out_features))
        self.bias_log_std = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x):
        weight_std = torch.exp(self.weight_log_std)
        bias_std = torch.exp(self.bias_log_std)
        
        # 重参数化采样
        weight = self.weight_mean + torch.randn_like(self.weight_mean) * weight_std
        bias = self.bias_mean + torch.randn_like(self.bias_mean) * bias_std
        
        return F.linear(self.dropout(x), weight, bias)


class BayesianNN(nn.Module):
    """贝叶斯神经网络 (使用 Dropout 近似)"""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list = [512, 256, 128],
        dropout: float = 0.3
    ):
        super().__init__()
        
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
        
        self.shared = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, output_dim)
        
        # Dropout 率用于测试时的蒙特卡洛采样
        self.dropout_rate = dropout
    
    def forward(self, x, sample=False):
        """
        Args:
            x: 输入特征
            sample: 是否进行 MC Dropout
        """
        if sample and self.training:
            # 训练时自然使用 dropout
            h = self.shared(x)
            return self.output(h)
        elif sample and not self.training:
            # MC Dropout: 多次采样
            h = F.dropout(self.shared(x), p=self.dropout_rate, training=True)
            return self.output(h)
        else:
            h = self.shared(x)
            return self.output(h)


class MCDropoutModel(nn.Module):
    """MC Dropout 模型 - 支持多次采样估计不确定性"""
    
    def __init__(self, base_model, n_samples=10):
        super().__init__()
        self.base_model = base_model
        self.n_samples = n_samples
    
    def forward(self, x):
        """多次采样返回预测和不确定性"""
        outputs = []
        for _ in range(self.n_samples):
            with torch.no_grad():
                output = self.base_model(x, sample=True)
                outputs.append(output)
        
        outputs = torch.stack(outputs)  # (n_samples, batch, n_classes)
        mean = outputs.mean(dim=0)
        std = outputs.std(dim=0)  # 不确定性
        
        return mean, std, outputs


class BayesianMultiTaskMLP(nn.Module):
    """多任务贝叶斯神经网络
    
    使用 MC Dropout 近似贝叶斯推断，支持:
    - 多任务联合学习 (EC + 定位 + 功能)
    - 不确定性估计
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dims: dict,  # {'ec': 8, 'loc': 11, 'func': 17}
        hidden_dims: list = [512, 256],
        dropout: float = 0.3,
        prior_std: float = 0.1
    ):
        super().__init__()
        
        self.output_dims = output_dims
        self.dropout_rate = dropout
        
        # 共享编码器 (使用 MC Dropout)
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
        
        # KL散度权重
        self.prior_std = prior_std
    
    def forward(self, x, sample=False):
        """前向传播
        
        Args:
            x: 输入特征 (batch, input_dim)
            sample: 是否使用 MC Dropout 采样
        
        Returns:
            dict: 各任务的logits
        """
        if sample:
            h = F.dropout(self.encoder(x), p=self.dropout_rate, training=True)
        else:
            h = self.encoder(x)
        
        return {
            task: head(h)
            for task, head in self.heads.items()
        }
    
    def mc_forward(self, x, n_samples=10):
        """蒙特卡洛前向传播，估算不确定性
        
        Args:
            x: 输入特征
            n_samples: 采样次数
        
        Returns:
            dict: {
                'mean': {task: mean_logits},
                'std': {task: uncertainty_std},
                'all_samples': {task: (n_samples, batch, n_classes)}
            }
        """
        self.eval()
        all_outputs = {task: [] for task in self.output_dims.keys()}
        
        with torch.no_grad():
            for _ in range(n_samples):
                outputs = self.forward(x, sample=True)
                for task, logits in outputs.items():
                    all_outputs[task].append(logits)
        
        results = {'mean': {}, 'std': {}, 'all_samples': {}}
        for task in self.output_dims.keys():
            stacked = torch.stack(all_outputs[task])  # (n_samples, batch, n_classes)
            results['mean'][task] = stacked.mean(dim=0)
            results['std'][task] = stacked.std(dim=0)
            results['all_samples'][task] = stacked
        
        return results
    
    def kl_divergence(self):
        """计算 KL 散度正则化项"""
        kl = 0
        for name, param in self.named_parameters():
            if 'weight' in name or 'bias' in name:
                kl += torch.sum(param ** 2) / (2 * self.prior_std ** 2)
        return kl
