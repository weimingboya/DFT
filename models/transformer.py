import torch
from torch import nn
import copy

from common.models.captioning_model import CaptioningModel
from common.models.containers import ModuleList

class Transformer(CaptioningModel):
    def __init__(self, bos_idx, encoder, decoder):
        super(Transformer, self).__init__()
        self.bos_idx = bos_idx
        self.encoder = encoder
        self.decoder = decoder
        self.register_state('grid_features', None)
        self.register_state('object_features', None)
        self.register_state('mask_enc', None)
        self.init_weights()

    @property
    def d_model(self):
        return self.decoder.d_model

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, images, seq, *args):
        if not isinstance(images, tuple) and not isinstance(images, list):
            images = [images]

        enc_output = self.encoder(*images)

        if not isinstance(enc_output, tuple) and not isinstance(enc_output, list):
            enc_output = [enc_output]
        if not isinstance(seq, tuple) and not isinstance(seq, list):
            seq = [seq]

        dec_output = self.decoder(*seq, *enc_output)
        return dec_output

    def init_state(self, b_s, device):
        return [torch.zeros((b_s, 0), dtype=torch.long, device=device),
                None, None]

    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        it = None
        if mode == 'teacher_forcing':
            raise NotImplementedError
        elif mode == 'feedback':
            if t == 0:
                if not isinstance(visual, tuple) and not isinstance(visual, list):
                    visual = [visual]
                grid_features, region_features = visual
                self.grid_features, self.object_features, self.mask_enc = self.encoder(grid_features, region_features)

                it = visual[0].data.new_full((visual[0].shape[0], 1), self.bos_idx).long() # (b_s,1)
            else:
                it = prev_output

        return self.decoder(it, self.grid_features, self.object_features, self.mask_enc)

class TransformerEnsemble(CaptioningModel):
    def __init__(self, model: Transformer, weight_files, device):
        super(TransformerEnsemble, self).__init__()
        self.n = len(weight_files)
        self.models = ModuleList([copy.deepcopy(model) for _ in range(self.n)])
        for i in range(self.n):
            state_dict_i = torch.load(weight_files[i], map_location=device)['state_dict']
            self.models[i].load_state_dict(state_dict_i)

    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        out_ensemble = []
        for i in range(self.n):
            out_i = self.models[i].step(t, prev_output, visual, seq, mode, **kwargs)
            out_ensemble.append(out_i.unsqueeze(0))

        return torch.mean(torch.cat(out_ensemble, 0), dim=0)
