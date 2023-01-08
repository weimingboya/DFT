import torch
import h5py
import os, pickle
from common.data.field import TextField
from models import build_encoder, build_decoder, Transformer, TransformerEnsemble

def visualize(model, dual_features, text_field, device):
    model.eval()
    with torch.no_grad():
        dual_features = [x.to(device) for x in dual_features]
        out, _ = model.beam_search(dual_features, 20, text_field.vocab.stoi['<eos>'], 1, out_size=1)
    
    caps_gen = text_field.decode(out, join_words=False)
    caps_gen = ' '.join(caps_gen[0]).strip()
    return caps_gen

def get_features_by_id(image_id, max_detections=50):
    clip_path = 'coco/features/COCO2014_RN50x4_GLOBAL.hdf5'
    vinvl_path = 'coco/features/COCO2014_VinVL.hdf5'
    clip_file = h5py.File(clip_path, 'r')
    vinvl_file = h5py.File(vinvl_path, 'r')

    feature_key = '%d_features' % image_id
    boxs_key = '%d_boxes' % image_id
    gird_feature = torch.from_numpy(clip_file[feature_key][()])
    region_feature = torch.from_numpy(vinvl_file[feature_key][()])
    boxes = torch.from_numpy(vinvl_file[boxs_key][()])

    delta = max_detections - region_feature.shape[0]
    if delta > 0:
        region_feature = torch.cat([region_feature, torch.zeros((delta, region_feature.shape[1]))], 0)
    elif delta < 0:
        region_feature = region_feature[:max_detections]

    return gird_feature, region_feature, boxes

def test():
    image_id = 108982
    device = 'cuda:0'
    # model_path = 'coco/checkpoints/DualModel/dual_add_best.pth'
    # model_path = 'coco/checkpoints/DualModel/dual_fuse_global_best.pth'
    model_path = 'coco/checkpoints/DualModel/dual_fuse_global5_rl_best.pth'

    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                            remove_punctuation=True, nopoints=False)
    vocab_path = 'cache/vocab.pkl'
    text_field.vocab = pickle.load(open(vocab_path, 'rb'))

    encoder = build_encoder(3, device=device)
    decoder = build_decoder(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)
    
    data = torch.load(model_path, map_location=device)
    model.load_state_dict(data['state_dict'])
    print(data['best_cider'])

    gird_feature, region_feature, boxes =  get_features_by_id(image_id)
    dual_features = [gird_feature.unsqueeze(0), region_feature.unsqueeze(0) ]
    caps_gen = visualize(model, dual_features, text_field, device)
    print(caps_gen)

