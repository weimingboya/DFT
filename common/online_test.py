import json
import random
import torch
from tqdm import tqdm
import argparse
import numpy as np
import os, pickle
from common.utils.utils import setup_seed
from common.data import OnlineTestDataset, DataLoader
from common.data.field import RawField, TextField, DualImageField
from models import build_encoder, build_decoder, Transformer, TransformerEnsemble

setup_seed(1234)
torch.backends.cudnn.benchmark = True

def online_test(model, dataloader, text_field, device):
    import itertools
    model.eval()
    result = []
    with tqdm(desc='Evaluation', unit='it', total=len(dataloader)) as pbar:
        for it, (images, image_ids) in enumerate(iter(dataloader)):
            with torch.no_grad():
                if isinstance(images, tuple) or isinstance(images, list):
                    images = [x.to(device) for x in images]
                else:
                    images = images.to(device)
                out, _ = model.beam_search(images, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)
            
            caps_gen = text_field.decode(out, join_words=False)
            for i, (image_id, gen_i) in enumerate(zip(image_ids, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                result.append({"image_id": image_id.item(), "caption": gen_i.strip()})
            pbar.update()
    return result

def test():
    parser = argparse.ArgumentParser(description='Dual Transformer')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model_path', type=str, default='coco/checkpoints/DualModel/dual_fuse_global_rl_best2.pth')
    parser.add_argument('--feature_type', type=str, default='clip')
    parser.add_argument('--features_path', type=str, default='coco/features/COCO2014_RN50x4_GLOBAL.hdf5')
    parser.add_argument('--image_folder', type=str, default='coco/images')
    parser.add_argument('--annotation_folder', type=str, default='coco/annotations')
    args = parser.parse_args()

    print('Dual Transformer Evaluation')

    # Pipeline for image regions
    # image_field = ImageDetectionsField(feature_type=args.feature_type, detections_path=args.features_path, max_detections=50)
    image_field = DualImageField(max_detections=50, global_feature=False, online_test=True)

    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)
    vocab_path = 'cache/vocab.pkl'
    text_field.vocab = pickle.load(open(vocab_path, 'rb'))

    ann_path = 'coco/annotations/image_info_test2014.json'
    # ann_path = 'coco/annotations/captions_val2014.json'
    dataset_test = OnlineTestDataset(ann_path, {'image': image_field, 'image_id': RawField()})
    dict_dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, drop_last=False)

    # Model and dataloaders
    encoder = build_encoder(3, device=args.device)
    decoder = build_decoder(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])

    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder)
    weight_files = ['coco/checkpoints/DualModel/dual_fuse_global_rl_best2.pth',
                    'coco/checkpoints/DualModel/dual_fuse_global2_rl_best.pth', 
                    'coco/checkpoints/DualModel/dual_fuse_global3_rl_best.pth', 
                    'coco/checkpoints/DualModel/dual_fuse_global5_rl_best.pth']
    model = TransformerEnsemble(model, weight_files, args.device).to(args.device)

    result = online_test(model, dict_dataloader_test, text_field, args.device)

    output_path = 'outputs/result/captions_test2014_DFT2_results.json'
    # output_path = 'outputs/result/captions_val2014_DFT2_results.json'
    with open(output_path, 'w') as fp:
        json.dump(result, fp)

    print("Online test result is over")