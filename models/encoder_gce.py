import torch
from torch import nn
from torch.nn import functional as F
from common.models.transformer.attention import MultiHeadAttention, NormSelfAttention
from common.models.transformer.utils import PolarRPE, PositionWiseFeedForward, RelationalEmbedding


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1):
        super(EncoderLayer, self).__init__()

        self.self_grid = MultiHeadAttention(d_model, d_k, d_v, h, dropout)

        self.self_region = MultiHeadAttention(d_model, d_k, d_v, h, dropout)
        self.global_grid = MultiHeadAttention(d_model, d_k, d_v, h, dropout, shortcut=False)
        self.global_region = MultiHeadAttention(d_model, d_k, d_v, h, dropout, shortcut=False)

        self.cls_grid = nn.Parameter(torch.randn(1, 1, d_model), requires_grad=True)
        self.cls_region = nn.Parameter(torch.randn(1, 1, d_model), requires_grad=True)
        
        self.pwff_grid = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.pwff_region = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, gird_features, region_features, attention_mask):
        b_s = region_features.shape[0]
        cls_grid = self.cls_grid.expand(b_s, 1, -1)
        cls_region = self.cls_region.expand(b_s, 1, -1)
        
        cls_grid = self.global_grid(cls_grid, gird_features, gird_features)
        cls_region = self.global_region(cls_region, region_features, region_features, attention_mask=attention_mask)
        
        gird_features = torch.cat([cls_region, gird_features], dim=1)
        region_features = torch.cat([cls_grid, region_features], dim=1)

        add_mask = torch.zeros(b_s, 1, 1, 1).bool().to(region_features.device)
        attention_mask = torch.cat([add_mask, attention_mask], dim=-1)
        grid_att = self.self_grid(gird_features, gird_features, gird_features)
        region_att = self.self_region(region_features, region_features, region_features, attention_mask=attention_mask)

        gird_ff = self.pwff_grid(grid_att)
        region_ff = self.pwff_region(region_att)

        gird_ff = gird_ff[:,1:]
        region_ff = region_ff[:,1:]

        return gird_ff, region_ff

class TransformerEncoder(nn.Module):
    def __init__(self, N, device='cuda', d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.device = device

        self.grid_proj = nn.Sequential(
            nn.Linear(2560, self.d_model),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.LayerNorm(self.d_model)
        )

        self.region_proj = nn.Sequential(
            nn.Linear(2048, self.d_model),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.LayerNorm(self.d_model)
        )

        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout) for _ in range(N)])

    def forward(self, grid_features, region_features):
        # input (b_s, seq_len)
        b_s = region_features.shape[0]
        attention_mask = (torch.sum(torch.abs(region_features), -1) == 0).unsqueeze(1).unsqueeze(1)
        grid_features = self.grid_proj(grid_features)
        region_features = self.region_proj(region_features)

        for l in self.layers:
            grid_features, region_features = l(grid_features, region_features, attention_mask)

        return grid_features, region_features, attention_mask

def build_encoder(N, device='cuda', d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1):
    Encoder = TransformerEncoder(N, device, d_model, d_k, d_v, h, d_ff, dropout)
    
    return Encoder