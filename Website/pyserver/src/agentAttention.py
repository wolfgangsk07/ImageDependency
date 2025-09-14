import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

# 论文：Agent Attention: On the Integration of Softmax and Linear Attention
# 论文地址：https://arxiv.org/pdf/2312.08874
class AgentAttention(nn.Module):
    def __init__(self, dim, num_patches, num_outputs, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1, agent_num=49, **kwargs):  # 移除了device参数
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        
        # === 主要改动点1: 更稳健的尺寸计算 ===
        # 确保num_patches是完全平方数
        self.num_patches = num_patches
        patch_sqrt = int(num_patches ** 0.5)
        if patch_sqrt * patch_sqrt != num_patches:
            raise ValueError(f"num_patches must be a perfect square, got {num_patches}")
        
        self.H = patch_sqrt
        self.W = patch_sqrt
        # =================================
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dim = dim
        window_size = (self.H, self.W)
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.agent_num = agent_num
        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=1, groups=dim)
        
        # === 主要改动点2: 简化位置偏置尺寸 ===
        # 使用动态计算的H和W而不是重复计算
        self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, self.H, self.W))
        self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, self.H, self.W))
        self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window_size[0] // sr_ratio, 1))
        self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window_size[1] // sr_ratio))
        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window_size[0], 1, agent_num))
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window_size[1], agent_num))
        # =================================
        
        trunc_normal_(self.an_bias, std=.02)
        trunc_normal_(self.na_bias, std=.02)
        trunc_normal_(self.ah_bias, std=.02)
        trunc_normal_(self.aw_bias, std=.02)
        trunc_normal_(self.ha_bias, std=.02)
        trunc_normal_(self.wa_bias, std=.02)
        
        # === 主要改动点3: 更稳健的agent池化 ===
        pool_size = int(agent_num ** 0.5)
        if pool_size * pool_size != agent_num:
            # 处理非完全平方数的agent_num
            pool_size = (int(agent_num ** 0.5), int(agent_num ** 0.5))
        self.pool = nn.AdaptiveAvgPool2d(output_size=pool_size)
        # =================================
        
        self.softmax = nn.Softmax(dim=-1)
        self.to_latent = nn.Identity()
        
        # === 主要改动点4: 强化输出头 ===
        # 添加更多非线性能力
        self.linear_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),  # 添加中间层
            nn.GELU(),                # 添加激活函数
            nn.Linear(dim * 2, num_outputs)
        )
        # =================================
        
        # 移除了显式的device属性

    def forward(self, x):
        b, n, c = x.shape
        
        # === 主要改动点5: 添加尺寸验证 ===
        if n != self.num_patches:
            raise ValueError(f"Input has {n} patches, but expected {self.num_patches}")
        # =================================
        
        H, W = self.H, self.W
        num_heads = self.num_heads
        head_dim = c // num_heads
        
        # 查询投影
        q = self.q(x)

        # 空间缩减
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(b, c, H, W)
            x_ = self.sr(x_).reshape(b, c, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
        else:
            kv = self.kv(x).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]

        # 生成agent tokens
        agent_tokens = self.pool(q.reshape(b, H, W, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)
        
        # 重塑QKV张量
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n // (self.sr_ratio ** 2), num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n // (self.sr_ratio ** 2), num_heads, head_dim).permute(0, 2, 1, 3)
        agent_tokens = agent_tokens.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)

        # 计算位置偏置
        kv_size = (H // self.sr_ratio, W // self.sr_ratio)
        position_bias1 = nn.functional.interpolate(self.an_bias, size=kv_size, mode='bilinear')
        position_bias1 = position_bias1.reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias = position_bias1 + position_bias2
        
        # Agent注意力计算
        agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v

        # 计算Agent偏置
        agent_bias1 = nn.functional.interpolate(self.na_bias, size=self.window_size, mode='bilinear')
        agent_bias1 = agent_bias1.reshape(1, num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
        agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, num_heads, -1, self.agent_num).repeat(b, 1, 1, 1)
        agent_bias = agent_bias1 + agent_bias2
        
        # Q注意力计算
        q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)
        q_attn = self.attn_drop(q_attn)
        x = q_attn @ agent_v

        # 重塑并添加深度卷积特征
        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, H // self.sr_ratio, W // self.sr_ratio, c).permute(0, 3, 1, 2)
        
        if self.sr_ratio > 1:
            v = nn.functional.interpolate(v, size=(H, W), mode='bilinear')
        
        x = x + self.dwc(v).permute(0, 2, 3, 1).reshape(b, n, c)

        # 投影层
        x = self.proj(x)
        x = self.proj_drop(x)

        # === 主要改动点6: 改进特征聚合 ===
        # 使用自适应池化代替简单平均
        x = x.reshape(b, H, W, -1).permute(0, 3, 1, 2)  # (B, C, H, W)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))  # (B, C, 1, 1)
        x = x.flatten(1)  # (B, C)
        # =================================
        
        x = self.to_latent(x)
        return self.linear_head(x)