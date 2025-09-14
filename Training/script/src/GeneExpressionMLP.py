import torch
import torch.nn as nn
import numpy as np
from timm.layers import DropPath, trunc_normal_
class GeneExpressionMLP(nn.Module):
    def __init__(self, dim, num_patches, num_outputs, hidden_dim=512, dropout=0.1, 
                 bias_init=None, num_hidden_layers=2, use_layer_norm=True, device="cuda"):
        """
        基因表达预测专用MLP
        
        参数:
            dim: 每个patch的输入特征维度
            num_patches: 每个样本的patch数量
            num_outputs: 输出基因数量
            hidden_dim: 隐藏层维度 (默认512)
            dropout: Dropout概率 (默认0.1)
            bias_init: 输出层偏置初始化值 (可选)
            num_hidden_layers: 隐藏层层数 (默认2)
            use_layer_norm: 是否使用LayerNorm (默认True)
            device: 计算设备 (默认"cuda")
        """
        super().__init__()
        self.dim = dim
        self.num_patches = num_patches
        self.num_outputs = num_outputs
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.device = device
        
        # 特征提取模块 (每个patch独立处理)
        layers = []
        input_dim = dim
        
        # 构建多个隐藏层
        for i in range(num_hidden_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            input_dim = hidden_dim  # 后续层的输入维度
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # 输出层 (每个基因一个输出)
        self.output_layer = nn.Linear(hidden_dim, num_outputs)
        
        # 初始化输出层偏置
        if bias_init is not None:
            if isinstance(bias_init, (int, float)):
                # 标量值初始化
                nn.init.constant_(self.output_layer.bias, bias_init)
            else:
                # 张量初始化
                assert bias_init.shape == (num_outputs,), "Bias init shape mismatch"
                self.output_layer.bias = nn.Parameter(bias_init.clone().detach())
        else:
            # 默认初始化为0
            nn.init.zeros_(self.output_layer.bias)
        
        self.to(device)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量，形状为 (batch_size, num_patches, dim)
            
        返回:
            基因表达预测值，形状为 (batch_size, num_outputs)
        """
        batch_size, num_patches, dim = x.shape
        
        # 验证输入维度
        if num_patches != self.num_patches:
            # 动态调整patch数量
            print(f"警告: 输入patch数量({num_patches})与初始化值({self.num_patches})不匹配，使用输入值")
            self.num_patches = num_patches
            
        # 处理每个patch: [batch_size, num_patches, dim] -> [batch_size, num_patches, hidden_dim]
        patch_features = self.feature_extractor(x)
        
        # 通过输出层: [batch_size, num_patches, hidden_dim] -> [batch_size, num_patches, num_outputs]
        gene_predictions = self.output_layer(patch_features)
        
        # 沿patches维度平均: [batch_size, num_patches, num_outputs] -> [batch_size, num_outputs]
        return gene_predictions.mean(dim=1)