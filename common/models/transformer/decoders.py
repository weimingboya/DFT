import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from .utils import PositionWiseFeedForward, sinusoid_encoding_table
from .attention import MultiHeadAttention
from ..containers import Module, ModuleList


class DecoderLayer(Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, self_att_module=None,
                 enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(DecoderLayer, self).__init__()
        self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True,
                                           attention_module=self_att_module,
                                           attention_module_kwargs=self_att_module_kwargs)
        self.enc_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
                                          attention_module=enc_att_module,
                                          attention_module_kwargs=enc_att_module_kwargs)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, input, enc_output, mask_pad, mask_self_att, mask_enc_att, pos=None):
        self_att = self.self_att(input, input, input, mask_self_att)
        self_att = self_att * mask_pad

        if pos is not None:
            enc_output = enc_output + pos

        enc_att = self.enc_att(self_att, enc_output, enc_output, mask_enc_att)
        enc_att = enc_att * mask_pad
        
        ff = self.pwff(enc_att)
        ff = ff * mask_pad
        return ff


class TransformerDecoder(Module):
    def __init__(self, vocab_size, max_len, N_dec, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, return_logits = False,
                 enc_dim=None, self_att_module=None, enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(TransformerDecoder, self).__init__()
        self.d_model = d_model
        self.pro_flag = False
        if enc_dim is not None and enc_dim != d_model:
            self.in_proj_model = nn.Sequential(
                nn.Linear(enc_dim, self.d_model),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.LayerNorm(self.d_model)
            )
            self.pro_flag = True

        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
        self.layers = ModuleList(
            [DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout, self_att_module=self_att_module,
                                enc_att_module=enc_att_module, self_att_module_kwargs=self_att_module_kwargs,
                                enc_att_module_kwargs=enc_att_module_kwargs) for _ in range(N_dec)])
        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        self.padding_idx = padding_idx
        self.N = N_dec
        self.return_logits = return_logits

        self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).byte())
        self.register_state('running_seq', torch.zeros((1,)).long())

    def forward(self, input, encoder_output, mask_encoder=None):
        if self.pro_flag:
            encoder_output = self.in_proj_model(encoder_output)

        # input (b_s, seq_len)
        b_s, seq_len = input.shape[:2]
        mask_queries = (input != self.padding_idx).unsqueeze(-1)  # (b_s, seq_len, 1)
        mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input.device),
                                         diagonal=1)
        mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        mask_self_attention = mask_self_attention + (input == self.padding_idx).unsqueeze(1).unsqueeze(1).byte()
        mask_self_attention = mask_self_attention.gt(0)  # (b_s, 1, seq_len, seq_len)

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)  # (b_s, seq_len)
        seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0)
        
        if self._is_stateful:
            self.running_mask_self_attention = torch.cat([self.running_mask_self_attention, mask_self_attention], -1)
            mask_self_attention = self.running_mask_self_attention
            self.running_seq.add_(1)
            seq = self.running_seq

        out = self.word_emb(input) + self.pos_emb(seq)
        for i, l in enumerate(self.layers):
            out = l(out, encoder_output, mask_queries, mask_self_attention, mask_encoder)

        out = self.fc(out)
        if self.return_logits:
            return out
        else:
            return F.log_softmax(out, dim=-1)