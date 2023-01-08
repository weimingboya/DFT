import math
import torch
from torch import nn

# 我提出的基于grid的极坐标相对位置编码
class PolarRPE(nn.Module):
    def __init__(self, k=3, h=8, d_k=64, d_r=256, window_size = (9, 9), device='cuda:0'):
        super(PolarRPE, self).__init__()
        Wh, Ww = window_size
        self.h = h
        self.d_k = d_k
        self.num_seq = Wh * Ww
        # num_direction = 4 * k + 1
        num_direction = 4 * k
        num_distance = math.floor(math.sqrt(Wh*Wh + Ww*Ww))

        # define a parameter table of relative position
        self.relative_table = nn.Embedding(num_direction * num_distance, d_r)
        self.projection = nn.Linear(d_r, h * d_k)

        # get pair-wise relative position index for each token inside the window
        coords_h, coords_w = torch.arange(Wh), torch.arange(Ww)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]), dim=-1)  # Wh, Ww, 2
        coords_flatten = coords.view(-1, 2)  # Wh*Ww, 2
        relative_coords = coords_flatten.unsqueeze(1) - coords_flatten.unsqueeze(0)  # Wh*Ww, Wh*Ww, 2
        relative_coords = relative_coords.view(-1, 2).float() # N*N, 2

        # relative_distance_pos
        norm_relative_distance = torch.norm(relative_coords, dim=-1)
        relative_distance_pos = norm_relative_distance.int()  # N*N

        # relative_direction_pos
        unit_direction_x = torch.cos(torch.arange(num_direction - 1) * math.pi / 2 / k)
        unit_direction_y = torch.sin(torch.arange(num_direction - 1) * math.pi / 2 / k)
        unit_direction = torch.stack([unit_direction_x, unit_direction_y])  # 2, 4k

        relative_direction = torch.matmul(relative_coords, unit_direction)
        relative_direction_pos = torch.argmax(relative_direction, dim=-1)  # N*N
        # relative_direction_pos = relative_direction_pos.masked_fill(norm_relative_distance == 0, num_direction-1)

        relative_pos = relative_direction_pos * num_distance + relative_distance_pos
        # relative_pos = relative_pos.masked_fill(norm_relative_distance == 0, num_direction * num_distance)

        self.relative_pos = relative_pos.to(device)

        self.init_weights()

    def init_weights(self):
        nn.init.uniform_(self.relative_table.weight, b=0.2)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.constant_(self.projection.bias, 0)

    def forward(self, bs):

        relative_emb = self.relative_table(self.relative_pos)
        relative_emb = self.projection(relative_emb).view(-1, self.h, self.d_k)  # (n*n, h, d_k)

        relative_emb = relative_emb.view(self.num_seq, self.num_seq, self.h, self.d_k).permute(2, 0, 1, 3)
        relative_emb = relative_emb.unsqueeze(0).expand(bs, self.h, self.num_seq, self.num_seq, self.d_k)  # (b_s, h, n, n, d_k)

        return relative_emb

if __name__ == '__main__':
    rpe = PolarRPE(device='cpu')
    rpe(2)







