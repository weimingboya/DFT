import numpy as np
import torch
from torch import nn
from einops import rearrange

from common.models.transformer.utils import Adapter, DWConv, MlpBlock
from ..containers import Module

class MultiHeadAttention(Module):
    '''
    Multi-head attention layer with Dropout and Layer Normalization.
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1, identity_map_reordering=False, can_be_stateful=False, 
                                        shortcut=True, attention_module=None, attention_module_kwargs=None):
        super(MultiHeadAttention, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.shortcut = shortcut
        if attention_module is not None:
            if attention_module_kwargs is not None:
                self.attention = attention_module(d_model=d_model, d_k=d_k, d_v=d_v, h=h, **attention_module_kwargs)
            else:
                self.attention = attention_module(d_model=d_model, d_k=d_k, d_v=d_v, h=h)
        else:
            self.attention = ScaledDotProductAttention(d_model=d_model, d_k=d_k, d_v=d_v, h=h)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.can_be_stateful = can_be_stateful
        if self.can_be_stateful:
            self.register_state('running_keys', torch.zeros((0, d_model)))
            self.register_state('running_values', torch.zeros((0, d_model)))

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys = self.running_keys

            self.running_values = torch.cat([self.running_values, values], 1)
            values = self.running_values

        if self.identity_map_reordering:
            q_norm = self.layer_norm(queries)
            k_norm = self.layer_norm(keys)
            v_norm = self.layer_norm(values)
            out = self.attention(q_norm, k_norm, v_norm, attention_mask, attention_weights)
            out = self.dropout(torch.relu(out))
            if self.shortcut:
                out = queries + out
        else:
            out = self.attention(queries, keys, values, attention_mask, attention_weights)
            out = self.dropout(out)
            if self.shortcut:
                out = queries + out
            out = self.layer_norm(out)
        return out


class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
            att = att + torch.log(torch.clamp(attention_weights, min=1e-6))
        if attention_mask is not None:
            att = att.masked_fill(attention_mask.bool(), -1e9)

        att = torch.softmax(att, -1)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


class ScaledDotProductAttentionMemory(nn.Module):
    '''
    Scaled dot-product attention with memory
    '''

    def __init__(self, d_model, d_k, d_v, h, m):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        :param m: Number of memory slots
        '''
        super(ScaledDotProductAttentionMemory, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.m_k = nn.Parameter(torch.FloatTensor(1, m, h * d_k))
        self.m_v = nn.Parameter(torch.FloatTensor(1, m, h * d_v))

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.m = m

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.normal_(self.m_k, 0, 1 / self.d_k)
        nn.init.normal_(self.m_v, 0, 1 / self.m)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        m_k = np.sqrt(self.d_k) * self.m_k.expand(b_s, self.m, self.h * self.d_k)
        m_v = np.sqrt(self.m) * self.m_v.expand(b_s, self.m, self.h * self.d_v)

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = torch.cat([self.fc_k(keys), m_k], 1).view(b_s, nk + self.m, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = torch.cat([self.fc_v(values), m_v], 1).view(b_s, nk + self.m, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = torch.cat([att[:, :, :, :nk] * attention_weights, att[:, :, :, nk:]], -1)
        if attention_mask is not None:
            att[:, :, :, :nk] = att[:, :, :, :nk].masked_fill(attention_mask.bool(), -1e9)
        att = torch.softmax(att, -1)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


class AoAttention(Module):
    '''
    Attention on attention module
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1):
        super(AoAttention, self).__init__()
        self.attention = ScaledDotProductAttention(d_model=d_model, d_k=d_k, d_v=d_v, h=h)
        self.aoa = nn.Sequential(nn.Linear(2 * d_model, 2 * d_model), nn.GLU())
        
        # self.dropout = nn.Dropout(p=dropout)
        # self.layer_norm = nn.LayerNorm(d_model)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.aoa[0].weight)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        out = self.attention(queries, keys, values, attention_mask, attention_weights)
        out = self.aoa(torch.cat([out, queries], -1))
        # out = self.dropout(out)
        # out = self.layer_norm(queries + out)
        return out


class LowRankAttention(Module):
    '''
    LowRankAttention module
    '''

    # def __init__(self, d_model=512, d_k=64, d_v=64, h=8, dropout=.1, can_be_stateful=True, mid_dim=128):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, dropout=.1, mid_dim=128, enable_norm=False ,with_pe=False):
        super(LowRankAttention, self).__init__()
        self.d_model = d_model
        self.enable_norm = enable_norm
        self.with_pe = with_pe
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.proj_attn_spatial = nn.Linear(mid_dim, 1)
        self.proj_attn_channel = nn.Linear(mid_dim, d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.in_proj_q1 = nn.Sequential(
            nn.Linear(d_model, h * d_k),
            nn.ELU(),
            nn.GroupNorm(h, h * d_k)
        )

        self.in_proj_q2 = nn.Sequential(
            nn.Linear(d_model, h * d_v),
            nn.ELU(),
            nn.GroupNorm(h, h * d_v)
        )

        self.in_proj_k = nn.Sequential(
            nn.Linear(d_model, h * d_k),
            nn.ELU(),
            nn.GroupNorm(h, h * d_k)
        )

        self.in_proj_v = nn.Sequential(
            nn.Linear(d_model, h * d_v),
            nn.ELU(),
            nn.GroupNorm(h, h * d_v)
        )

        self.proj_attn_map = nn.Sequential(
            nn.Linear(d_k, mid_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        if enable_norm:
            self.norm = nn.InstanceNorm1d(d_k)

        if with_pe:
            self.fc_gq = nn.Linear(d_model, h * d_k)

        # self.can_be_stateful = can_be_stateful
        # if self.can_be_stateful:
        #     self.register_state('running_keys', torch.zeros((0, d_model)))
        #     self.register_state('running_values', torch.zeros((0, d_model)))

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, queries, keys, values, attention_mask=None, geometric_attention=None):
        # if self.can_be_stateful and self._is_stateful:
        #     self.running_keys = torch.cat([self.running_keys, keys], 1)
        #     keys = self.running_keys

        #     self.running_values = torch.cat([self.running_values, values], 1)
        #     values = self.running_values

        bs, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = queries.view(bs * nq, -1)
        if self.enable_norm:
            q1 = self.in_proj_q1(q).view(bs, nq, self.h, self.d_k).permute(0, 2, 3, 1) \
                              .contiguous().view(bs * self.h, self.d_k, nq)  # (b_s*h, d_k, nq)
            q2 = self.in_proj_q2(q).view(bs, nq, self.h, self.d_v).permute(0, 2, 3, 1) \
                              .contiguous().view(bs * self.h, self.d_k, nq)  # (b_s*h, d_k, nq)

            q1 = self.norm(q1).view(bs, self.h, self.d_k, nq).permute(0, 1, 3, 2)  # (b_s, h, nq, d_k)
            q2 = self.norm(q2).view(bs, self.h, self.d_k, nq).permute(0, 1, 3, 2)  # (b_s, h, nq, d_k)
        else:
            q1 = self.in_proj_q1(q).view(bs, nq, self.h, self.d_k).transpose(1, 2)
            q2 = self.in_proj_q2(q).view(bs, nq, self.h, self.d_v).transpose(1, 2)

        k = keys.view(bs * nk, -1)
        v = values.view(bs * nk, -1)
        k = self.in_proj_k(k).view(bs, nk, self.h, self.d_k).transpose(1, 2)
        v = self.in_proj_v(v).view(bs, nk, self.h, self.d_v).transpose(1, 2)

        attn_map = self.proj_attn_map(q1.unsqueeze(-2) * k.unsqueeze(-3))

        attn_spatial = self.proj_attn_spatial(attn_map).squeeze(-1)

        if self.with_pe:
            assert geometric_attention is not None
            gq = self.fc_gq(queries).view(bs, nq, self.h, self.d_k).permute(0, 2, 1, 3).unsqueeze(-1)  # (b_s, h, nq, d_k, 1)
            geometric_bias = torch.matmul(geometric_attention, gq).squeeze(-1)  # (b_s, h, nq, nk)
            attn_spatial = attn_spatial + geometric_bias

        if attention_mask is not None:
            attn_spatial = attn_spatial.masked_fill(attention_mask.bool(), -1e9)
            att_mask_ext = ~attention_mask.unsqueeze(-1)
            att_map_pool = torch.sum(attn_map * att_mask_ext, -2) / torch.sum(att_mask_ext, -2)
        else:
            att_map_pool = attn_map.mean(-2)

        attn_spatial = torch.softmax(attn_spatial, dim=-1)
        attn_channel = torch.sigmoid(self.proj_attn_channel(att_map_pool))

        attn_v = torch.matmul(attn_spatial, v) * q2 * attn_channel

        out = attn_v.transpose(1, 2).contiguous().view(bs, -1, self.h * self.d_v)

        # out = self.dropout(self.fc_o(out))
        # out = self.layer_norm(queries + out)
        out = self.fc_o(out)
        return out


class NormSelfAttention(nn.Module):
    '''
    Normalized Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h, enable_norm=False, with_pe=None):
        super(NormSelfAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)

        if enable_norm:
            self.norm = nn.InstanceNorm1d(d_k)

        if with_pe and with_pe == 'rpe':
            self.fc_gq = nn.Linear(d_model, h * d_k)

        self.d_model = d_model
        self.enable_norm = enable_norm
        self.with_pe = with_pe
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        if self.enable_norm:
            q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 3, 1) \
                              .contiguous().view(b_s * self.h, self.d_k, nq)  # (b_s*h, d_k, nq)
            q = self.norm(q).view(b_s, self.h, self.d_k, nq).permute(0, 1, 3, 2)  # (b_s, h, nq, d_k)
        else:
            q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)

        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if self.with_pe and self.with_pe == 'rpe':
            assert attention_weights is not None
            gq = self.fc_gq(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3).unsqueeze(-1)  # (b_s, h, nq, d_k, 1)
            geometric_bias = torch.matmul(attention_weights, gq).squeeze(-1)  # (b_s, h, nq, nk)
            att = att + geometric_bias
            # att = att + attention_weights
            # att = att + torch.log(torch.clamp(attention_weights, min=1e-6))

        if attention_mask is not None:
            att = att.masked_fill(attention_mask.bool(), -1e9)
        att = torch.softmax(att, -1)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


class ScaledDotProductAdaptiveAttention(nn.Module):
    '''
    Scaled dot-product attention with Language
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1):
        super(ScaledDotProductAdaptiveAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.fc_s = nn.Linear(d_model, h * d_k)

        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.xavier_uniform_(self.fc_s.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)
        nn.init.constant_(self.fc_s.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None, language_feature=None):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        s = self.fc_s(language_feature).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)

        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        # 视觉
        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -1e9)
        # 语言
        # language_att = torch.matmul(q, s.permute(0, 1, 3, 2)) / np.sqrt(self.d_k)  # (b_s, h, nq, nq)
        # language_att = torch.cat([language_att[:, :, i, i].unsqueeze(-1) for i in range(nq)], -1) # (b_s, h, nq)
        language_att = torch.cat([q[:,:,i,:].unsqueeze(-2) @ s[:,:,i,:].unsqueeze(-1) / np.sqrt(self.d_k) for i in range(nq)], -2)

        # 融合att
        # combined_att = torch.cat([att, language_att.unsqueeze(-1)], -1)     # (b_s, h, nq, nk + 1)
        # combined_att = [torch.softmax(combined_att[:, :, i, :].unsqueeze(2), -1) for i in range(nq)]
        # # dropout
        # combined_att = [self.dropout(item) for item in combined_att]
        combined_att = torch.cat([att, language_att], -1)
        combined_att = torch.softmax(combined_att, -1) # (b_s, h, nq, nk + 1)

        # 融合v
        # combined_v = [torch.cat([v, s[:, :, i, :].unsqueeze(2)], 2) for i in range(nq)]
        # assert len(combined_att) == len(combined_v) == nq
        # out = torch.cat([torch.matmul(combined_att[i], combined_v[i]) for i in range(nq)], 2)
        att_v = torch.matmul(att, v)
        beta = combined_att[:,:,:,-1].unsqueeze(-1)
        out = beta * s + (1 - beta) * att_v

        out = out.permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


class MemoryAttention(nn.Module):
    '''
    Scaled dot-product attention with all memory
    '''

    def __init__(self, d_model, d_k, d_v, h, m=80):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        :param m: Number of memory slots
        '''
        super(MemoryAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.m_k = nn.Parameter(torch.FloatTensor(1, m, h * d_k))
        self.m_v = nn.Parameter(torch.FloatTensor(1, m, h * d_v))

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.m = m

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)
        nn.init.normal_(self.m_k, 0, 1 / self.d_k)
        nn.init.normal_(self.m_v, 0, 1 / self.m)

    def forward(self, queries, keys, values, attention_mask=None, geometric_attention=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]

        m_k = self.m_k.expand(b_s, self.m, self.h * self.d_k) * np.sqrt(self.d_k)
        m_v = self.m_v.expand(b_s, self.m, self.h * self.d_v) * np.sqrt(self.m)

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = m_k.view(b_s, self.m, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, m)
        v = m_v.view(b_s, self.m, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, m, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, m)
        # if attention_mask is not None:
        #     att = att.masked_fill(attention_mask.bool(), -1e9)
        att = torch.softmax(att, -1)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


class AdapterAttention(nn.Module):

    def __init__(self, d_model, d_k, d_v, h, dropout=.1,  mid_dim=40, with_pe=None):

        super(AdapterAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)

        # self.lamda = nn.Sequential(
        #     nn.Linear(d_model * 2, 1),
        #     nn.Sigmoid()
        #     )
        self.aoa1 = nn.Sequential(nn.Linear(2 * d_model, 2 * d_model), nn.GLU())
        # self.aoa2 = nn.Sequential(nn.Linear(2 * d_model, 2 * d_model), nn.GLU())

        self.adapter = Adapter(d_model, d_v, h, mid_dim, dropout)
        # self.dc_conv = DWConv(d_model)
        # self.mlp = MlpBlock(in_dim=81, mlp_dim=256)

        if with_pe and with_pe == 'rpe':
            self.fc_gq = nn.Linear(d_model, h * d_k)

        self.d_model = d_model
        self.with_pe = with_pe
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if self.with_pe and self.with_pe == 'rpe':
            assert attention_weights is not None
            gq = self.fc_gq(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3).unsqueeze(-1)  # (b_s, h, nq, d_k, 1)
            geometric_bias = torch.matmul(attention_weights, gq).squeeze(-1)  # (b_s, h, nq, nk)
            att = att + geometric_bias

        if attention_mask is not None:
            att = att.masked_fill(attention_mask.bool(), -1e9)
        att = torch.softmax(att, -1)

        # 1) adapter: mid_dim=32 CIDEr 122.89 (head add)
        # 2) val: mid_dim=32 CIDEr 122.16 (lamda head add)
        # 3) adapter2: mid_dim=40 CIDEr 123.11（lamda add）

        h = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        h = self.fc_o(h)  # (b_s, nq, d_model)

        delta_h = self.adapter(values)
        # delta_h = self.dc_conv(values)
        # delta_h = self.mlp(values)
        
        # x = torch.cat([h, delta_h], dim=-1)
        # lamda = self.lamda(x)
        # out = (1 - lamda) * h + lamda * delta_h

        out = self.aoa1(torch.cat([h, delta_h], dim=-1))

        # h = self.aoa1(torch.cat([h, queries], dim=-1))
        # delta_h = self.aoa2(torch.cat([delta_h, queries], dim=-1))
        # out = h + delta_h

        return out


class RPEAttention(nn.Module):
    '''
    Normalized Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h):
        super(RPEAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)

        self.fc_gq = nn.Linear(d_model, h * d_k)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)

        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        gq = self.fc_gq(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3).unsqueeze(-1)  # (b_s, h, nq, d_k, 1)
        geometric_bias = torch.matmul(attention_weights, gq).squeeze(-1)  # (b_s, h, nq, nk)
        att = att + geometric_bias

        # k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nk, d_k)
        # q = q.unsqueeze(-2).expand(b_s, self.h, nq, nk, self.d_k)  # (b_s, h, nq, nk, d_k)
        # k = k.unsqueeze(-3).expand(b_s, self.h, nq, nk, self.d_k)  # (b_s, h, nq, nk, d_k)
        # q = q + attention_weights
        # att = torch.sum(q * k, dim=-1) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)

        if attention_mask is not None:
            att = att.masked_fill(attention_mask.bool(), -1e9)
        att = torch.softmax(att, -1)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


class ChannelAttention(nn.Module):

    def __init__(self, d_model, num_group, dropout=.1, qkv_bias=False):
        super().__init__()
        self.num_group = num_group
        head_dim = d_model // num_group
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(d_model, d_model * 3, bias=qkv_bias)
        self.proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)


    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_group, C // self.num_group).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        k = k * self.scale
        attention = k.transpose(-1, -2) @ v
        attention = attention.softmax(dim=-1)
        out = (attention @ q.transpose(-1, -2)).transpose(-1, -2)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)

        out = self.dropout(out)
        out = self.layer_norm(x + out)
        return out


class OSAttention(nn.Module):
    '''
    Normalized Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h):
        super(OSAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        queries: (bs, n_q, d_model)
        keys: (bs, n_q, n_k, d_model)
        values: (bs, n_q, n_k, d_model)
        '''

        b_s, nq = queries.shape[:2]
        nk = keys.shape[-2]

        q = self.fc_q(queries)  # (b_s, nq, h*d_k)
        q = rearrange(q, 'b q (h d) -> b h q d', h=self.h, d=self.d_k)  # (b_s, h, nq, d_k)
        q = q.unsqueeze(-2).expand(b_s, self.h, nq, nk, self.d_k)  # (b_s, h, nq, nk, d_k)

        k = self.fc_k(keys.view(b_s, nq*nk, -1)) # (b_s, nq*nk, h*d_k)
        v = self.fc_v(values.view(b_s, nq*nk, -1)) # (b_s, nq*nk, h*d_v)
        k = rearrange(k, 'b (q k) (h d) -> b h q k d', q=nq, k=nk, h=self.h, d=self.d_k) # (b_s, h, nq, nk, d_k)
        v = rearrange(v, 'b (q k) (h d) -> b h q k d', q=nq, k=nk, h=self.h, d=self.d_v) # (b_s, h, nq, nk, d_v)

        att = torch.sum(q * k, dim=-1) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)

        if attention_mask is not None:
            att = att.masked_fill(attention_mask.bool(), -1e9)

        att = torch.softmax(att, -1)  # (b_s, h, nq, nk)
        att = att.unsqueeze(-1).expand(*att.shape, self.d_v) # (b_s, h, nq, nk, d_v)
        out = torch.sum(att * v, dim=-2) # (b_s, h, nq, d_v)
        out = rearrange(out, 'b h q d -> b q (h d)')  # (b_s, nq, h*d_v)

        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out