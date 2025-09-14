import torch.nn as nn
import torch
from einops import rearrange
from huggingface_hub import PyTorchModelHubMixin
from timm.layers import trunc_normal_

class SummaryMixing(nn.Module):
    def __init__(self, input_dim, dimensions_f, dimensions_s, dimensions_c):
        super().__init__()
        
        self.local_norm = nn.LayerNorm(dimensions_f)
        self.summary_norm = nn.LayerNorm(dimensions_s)

        self.s = nn.Linear(input_dim, dimensions_s)
        self.f = nn.Linear(input_dim, dimensions_f)
        self.c = nn.Linear(dimensions_s + dimensions_f, dimensions_c)

    def forward(self, x):

        local_summ = torch.nn.GELU()(self.local_norm(self.f(x)))
        time_summ = self.s(x)    
        time_summ = torch.nn.GELU()(self.summary_norm(torch.mean(time_summ, dim=1)))
        time_summ = time_summ.unsqueeze(1).repeat(1, x.shape[1], 1)
        out = torch.nn.GELU()(self.c(torch.cat([local_summ, time_summ], dim=-1)))

        return out
    

class MultiHeadSummary(nn.Module):
    def __init__(self, nheads, input_dim, dimensions_f, dimensions_s, dimensions_c, dimensions_projection):
        super().__init__()

        self.mixers = nn.ModuleList([])
        for _ in range(nheads):
            self.mixers.append(SummaryMixing(input_dim=input_dim, dimensions_f=dimensions_f, dimensions_s=dimensions_s, dimensions_c=dimensions_c))

        self.projection = nn.Linear(nheads*dimensions_c, dimensions_projection)

    def forward(self, x):

        outs = []
        for mixer in self.mixers:
            outs.append(mixer(x))
        
        outs = torch.cat(outs, dim=-1)
        out = self.projection(outs)

        return out
    

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)


class SummaryTransformer(nn.Module):
    def __init__(self, input_dim, depth, nheads, dimensions_f, dimensions_s, dimensions_c):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                MultiHeadSummary(nheads, input_dim, dimensions_f, dimensions_s, dimensions_c, dimensions_projection=input_dim),
                FeedForward(input_dim, input_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x                 # dimensions_projection needs to be equal to input_dim
            x = ff(x) + x                   # output_dim of feedforward needs to be equal to input_dim
        return x


class Tformer_lin(nn.Module, PyTorchModelHubMixin):
    def __init__(self, num_outputs, input_dim, depth, nheads, 
                        dimensions_f, dimensions_s, dimensions_c, 
                        num_clusters=900):
        super().__init__()

        self.pos_emb1D = nn.Parameter(torch.randn(num_clusters, input_dim))
        trunc_normal_(self.pos_emb1D, std=.02)  # 添加初始化
        self.transformer = SummaryTransformer(input_dim, depth, nheads, dimensions_f, dimensions_s, dimensions_c)

        self.to_latent = nn.Identity()
        self.linear_head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, num_outputs)
        )

    def forward(self, x):
        # 确保位置嵌入与输入在同一设备上
        pos_emb = self.pos_emb1D.to(x.device)
        #pe = posemb_sincos_2d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pos_emb  # 使用移动后的位置嵌入

        x = self.transformer(x)
        x = x.mean(dim = 1)

        x = self.to_latent(x)
        return self.linear_head(x)
    def to(self, device=None, dtype=None, non_blocking=False):
        # 重写 to 方法确保正确移动所有参数
        self = super().to(device=device, dtype=dtype, non_blocking=non_blocking)
        self.pos_emb1D = nn.Parameter(self.pos_emb1D.to(device))
        return self