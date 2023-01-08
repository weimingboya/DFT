import math
import torch
from torch import nn
import torch.nn.functional as F

# scale the pos to (0 ~ 1)
def get_relative_pos(x, batch_size, norm_len):
    x = x.view(1, -1, 1).expand(batch_size, -1, -1)
    return  x / norm_len

def get_grids_pos(batch_size, seq_len, grid_size=(7, 7), device='cuda'):
    assert seq_len == grid_size[0] * grid_size[1]

    # record the pos of each grid according to the form of region box
    x = torch.arange(0, grid_size[0]).float().to(device)
    y = torch.arange(0, grid_size[1]).float().to(device)

    px_min = x.view(-1, 1).expand(-1, grid_size[0]).contiguous().view(-1)
    py_min = y.view(1, -1).expand(grid_size[1], -1).contiguous().view(-1)

    px_max = px_min + 1
    py_max = py_min + 1

    # scale pos
    rpx_min = get_relative_pos(px_min, batch_size, grid_size[0])
    rpy_min = get_relative_pos(py_min, batch_size, grid_size[1])

    rpx_max = get_relative_pos(px_max, batch_size, grid_size[0])
    rpy_max = get_relative_pos(py_max, batch_size, grid_size[1])

    return rpx_min, rpy_min, rpx_max, rpy_max

# 适用于bbox或者grid网格的相对位置编码
def RelationalEmbedding(f_g, dim_g=64, wave_len=1000, is_gird=False, trignometric_embedding=True):
    """
    Given a tensor with bbox coordinates for detected objects on each batch image,
    this function computes a matrix for each image
    with entry (i,j) given by a vector representation of the
    displacement between the coordinates of bbox_i, and bbox_j
    input: np.array of shape=(batch_size, max_nr_bounding_boxes, 4)
    output: np.array of shape=(batch_size, max_nr_bounding_boxes, max_nr_bounding_boxes, 64)
    """
    # returns a relational embedding for each pair of bboxes, with dimension = dim_g
    # follow implementation of https://github.com/heefe92/Relation_Networks-pytorch/blob/master/model.py#L1014-L1055
    device = f_g.device

    if is_gird:
        batch_size, seq_len = f_g.shape[:2]
        gs = int(math.sqrt(seq_len))
        x_min, y_min, x_max, y_max = get_grids_pos(batch_size, seq_len, grid_size=(gs, gs), device=device)
    else:
        batch_size = f_g.size(0)
        x_min, y_min, x_max, y_max = torch.chunk(f_g, 4, dim=-1)

    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    w = (x_max - x_min) + 1.
    h = (y_max - y_min) + 1.

    # cx.view(1,-1) transposes the vector cx, and so dim(delta_x) = (dim(cx), dim(cx))
    delta_x = cx - cx.view(batch_size, 1, -1)
    delta_x = torch.clamp(torch.abs(delta_x / w), min=1e-3)
    delta_x = torch.log(delta_x)

    delta_y = cy - cy.view(batch_size, 1, -1)
    delta_y = torch.clamp(torch.abs(delta_y / h), min=1e-3)
    delta_y = torch.log(delta_y)

    delta_w = torch.log(w / w.view(batch_size, 1, -1))
    delta_h = torch.log(h / h.view(batch_size, 1, -1))

    matrix_size = delta_h.size()
    delta_x = delta_x.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_y = delta_y.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_w = delta_w.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_h = delta_h.view(batch_size, matrix_size[1], matrix_size[2], 1)

    position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h), -1)  # bs * r * r * 4

    if trignometric_embedding == True:
        feat_range = torch.arange(dim_g / 8).to(device)
        dim_mat = feat_range / (dim_g / 8)
        dim_mat = 1. / (torch.pow(wave_len, dim_mat))

        dim_mat = dim_mat.view(1, 1, 1, -1)
        position_mat = position_mat.view(batch_size, matrix_size[1], matrix_size[2], 4, -1)
        position_mat = 100. * position_mat

        mul_mat = position_mat * dim_mat
        mul_mat = mul_mat.view(batch_size, matrix_size[1], matrix_size[2], -1)
        sin_mat = torch.sin(mul_mat)
        cos_mat = torch.cos(mul_mat)
        embedding = torch.cat((sin_mat, cos_mat), -1)
    else:
        embedding = position_mat
    return embedding


def position_embedding(input, d_model):
    device = input.device
    input = input.view(-1, 1)
    dim = torch.arange(d_model // 2, dtype=torch.float32, device=device).view(1, -1)
    sin = torch.sin(input / 10000 ** (2 * dim / d_model))
    cos = torch.cos(input / 10000 ** (2 * dim / d_model))

    out = torch.zeros((input.shape[0], d_model), device=device)
    out[:, ::2] = sin
    out[:, 1::2] = cos
    return out

# sin cos绝对位置编码
def sinusoid_encoding_table(max_len, d_model, padding_idx=None):
    pos = torch.arange(max_len, dtype=torch.float32)
    out = position_embedding(pos, d_model)

    if padding_idx is not None:
        out[padding_idx] = 0
    return out

# 基于Grid网格的sin cos绝对位置编码
class GridPESine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        device = x.device 
        if mask is None:
            mask = torch.zeros(x.shape[:-1], dtype=torch.bool, device=device)
        not_mask = (mask == False)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3)  # .permute(0, 3, 1, 2)
        pos = pos.flatten(1, 2)
        return pos


class PositionWiseFeedForward(nn.Module):
    '''
    Position-wise feed forward layer
    '''

    def __init__(self, d_model=512, d_ff=2048, dropout=.1, act_fn='ReLU', identity_map_reordering=False, local=False):
        super(PositionWiseFeedForward, self).__init__()
        self.local = local
        self.identity_map_reordering = identity_map_reordering
        if local:
            self.dwconv = DWConv(d_ff, gird_size=(9, 9))
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.act = getattr(nn, act_fn)()

    def forward(self, input):
        if self.identity_map_reordering:
            x = self.layer_norm(input)
            x = self.fc1(x)
            if self.local:
                x = x + self.dwconv(x)
            x = self.act(x)
            x = self.dropout_2(x)
            x = self.fc2(x)
            x = input + self.dropout(self.act(x))
        else:
            x = self.fc1(input)
            if self.local:
                x = self.dwconv(x)
            x = self.act(x)
            x = self.dropout_2(x)
            x = self.fc2(x)
            x = self.dropout(x)
            x = self.layer_norm(input + x)
        return x


class FFNWithPrivateLN(nn.Module):
    '''
    Position-wise feed forward layer
    '''

    def __init__(self, d_model=512, d_ff=2048, dropout=.1):
        super(FFNWithPrivateLN, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, input, m=0):
        out = self.fc2(self.dropout_2(F.relu(self.fc1(input))))
        out = self.dropout(out)
        if m == 0:
            out = self.layer_norm(input + out)
        elif m == 1:
            out = self.layer_norm1(input + out)
        else:
            out = self.layer_norm2(input + out)
        return out


class LocalFeedForward(nn.Module):

    def __init__(self, d_model=512, d_ff=2048, dropout=.1):
        super(PositionWiseFeedForward, self).__init__()
        self.dwconv = DWConv(d_ff, gird_size=(9, 9))
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.act = nn.ReLU()

    def forward(self, input):
        x = self.fc1(input)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.dropout_2(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.layer_norm(input + x)
        return x


class Adapter(nn.Module):
    def __init__(self, d_model=512, d_v=64, h=8, mid_dim=40, dropout=.1, act_fn='ReLU'):
        super(Adapter, self).__init__()

        self.fc_dalta_o = nn.Linear(h * d_v, d_model)

        self.mh_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_v),
                # nn.Linear(d_v, mid_dim),
                # getattr(nn, act_fn)(),
                # nn.Linear(mid_dim, d_v)
                DWConv(d_v, gird_size=(9, 9)),
                # nn.ReLU(),
                # DWConv(d_v, gird_size=(9, 9))
                )
            for _ in range(h)])

        # self.act = nn.ReLU()
        # self.dropout = nn.Dropout(p=dropout)
        # self.layer_norm = nn.LayerNorm(d_model)

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, input):

        delta_hs = [l(input) for l in self.mh_adapters]
        delta_h = torch.cat(delta_hs, dim=-1)  # (b_s, nq, h*d_v)
        delta_h = self.fc_dalta_o(delta_h)

        # delta_h = self.act(delta_h)

        # delta_h = self.dropout(delta_h)
        # delta_h = self.layer_norm(input + delta_h)
        # delta_h = input + delta_h

        return delta_h


class DWConv(nn.Module):
    def __init__(self, dim=64, gird_size=(9, 9)):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.gird_size = gird_size
        self.act = nn.ReLU()

        # self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.dwconv.weight)

    def forward(self, x):
        B, N, C = x.shape
        H, W = self.gird_size
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.act(x)

        return x

class MlpBlock(nn.Module):
    def __init__(self, in_dim, mlp_dim):
        super(MlpBlock, self).__init__()
        self.ff1 = nn.Linear(in_dim, mlp_dim)
        self.act = nn.ReLU()
        self.ff2 = nn.Linear(mlp_dim, in_dim)

    def __call__(self, x):
        x = x.transpose(1, 2)
        x = self.ff1(x)
        x = self.act(x)
        x = self.ff2(x)
        x = x.transpose(1, 2)
        return x

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
        # self.relative_direction_table = nn.Embedding(num_direction, d_r)
        # self.relative_distance_table = nn.Embedding(num_distance, d_r)
        self.relative_table = nn.Embedding(num_direction * num_distance, d_r)
        self.projection = nn.Linear(d_r, h * d_k)
        # self.projection = nn.Linear(d_r, h)
        # self.act = nn.ReLU()
        # self.projection = nn.Linear(d_r * 2, h * d_k)

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

        # self.relative_direction_pos = relative_direction_pos.to(device)
        # self.relative_distance_pos = relative_distance_pos.to(device)
        self.relative_pos = relative_pos.to(device)

        self.init_weights()

    def init_weights(self):
        # nn.init.uniform_(self.relative_direction_table.weight, b=0.2)
        # nn.init.uniform_(self.relative_distance_table.weight, b=0.2)
        nn.init.uniform_(self.relative_table.weight, b=0.2)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.constant_(self.projection.bias, 0)

    def forward(self, bs):

        # direction + distance
        # relative_direction_emb = self.relative_direction_table(self.relative_direction_pos)
        # relative_distance_emb = self.relative_distance_table(self.relative_distance_pos)
        # relative_emb = relative_direction_emb + relative_distance_emb # (n*n, d_r)

        # relative_emb = torch.cat([relative_direction_emb, relative_distance_emb], dim=-1)  # (n*n, d_r * 2)
        relative_emb = self.relative_table(self.relative_pos)
        relative_emb = self.projection(relative_emb).view(-1, self.h, self.d_k)  # (n*n, h, d_k)

        # relative_emb = self.projection(relative_emb)  # (n*n, h)
        # relative_emb = self.act(relative_emb)

        # direction
        # relative_direction_emb = self.relative_direction_table(self.relative_direction_pos) # (n*n, d_r)
        # relative_emb = self.projection(relative_direction_emb).view(-1, self.h, self.d_k)  # (n*n, h, d_k)

        # distance
        # relative_distance_emb = self.relative_distance_table(self.relative_distance_pos) # (n*n, d_r)
        # relative_emb = self.projection(relative_distance_emb).view(-1, self.h, self.d_k)  # (n*n, h, d_k)

        relative_emb = relative_emb.view(self.num_seq, self.num_seq, self.h, self.d_k).permute(2, 0, 1, 3)
        relative_emb = relative_emb.unsqueeze(0).expand(bs, self.h, self.num_seq, self.num_seq, self.d_k)  # (b_s, h, n, n, d_k)
        
        # relative_emb = relative_emb.view(self.num_seq, self.num_seq, self.h).permute(2, 0, 1)
        # relative_emb = relative_emb.unsqueeze(0).expand(bs, self.h, self.num_seq, self.num_seq)  # (b_s, h, n, n)

        return relative_emb

if __name__ == '__main__':
    rpe = PolarRPE(device='cpu')
    rpe(2)







