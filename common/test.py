import random
import torch
from tqdm import tqdm
import argparse
import numpy as np
import common.evaluation as evaluation
from common.utils.utils import create_dataset, setup_seed
from common.data import DataLoader
from common.data.field import RawField, TextField, DualImageField
from models import build_encoder, build_decoder, Transformer, TransformerEnsemble

setup_seed(1234)
torch.backends.cudnn.benchmark = True

def predict_captions(model, dataloader, text_field, device):
    import itertools
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='Evaluation', unit='it', total=len(dataloader)) as pbar:
        for it, (images, caps_gt) in enumerate(iter(dataloader)):
            with torch.no_grad():
                if isinstance(images, tuple) or isinstance(images, list):
                    images = [x.to(device) for x in images]
                else:
                    images = images.to(device)
                out, _ = model.beam_search(images, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)
            
            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i.strip(), ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    return scores


def test():
    parser = argparse.ArgumentParser(description='Dual Transformer')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model_path', type=str, default='Camel/saved_models/Camel_best.pth')
    parser.add_argument('--feature_type', type=str, default='clip')
    parser.add_argument('--features_path', type=str, default='coco/features/COCO2014_RN50x4_GLOBAL.hdf5')
    parser.add_argument('--image_folder', type=str, default='coco/images')
    parser.add_argument('--annotation_folder', type=str, default='coco/annotations')
    args = parser.parse_args()

    print('Dual Transformer Evaluation')

    # Pipeline for image regions
    image_field = DualImageField(max_detections=50, global_feature=False)

    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)

    # Create the dataset
    datasets = create_dataset(args, image_field, text_field)
    _, val_dataset, test_dataset = datasets

    dict_dataset_val = val_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataloader_val = DataLoader(dict_dataset_val, batch_size=args.batch_size // 5, num_workers=args.workers, pin_memory=True, drop_last=False)
    dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size // 5, num_workers=args.workers, pin_memory=True, drop_last=False)

    # Model and dataloaders
    encoder = build_encoder(3, device=args.device)
    decoder = build_decoder(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])

    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(args.device)

    # weight_files = ['coco/checkpoints/DualModel/dual_fuse_global_rl_best2.pth',
    #                 'coco/checkpoints/DualModel/dual_fuse_global2_rl_best.pth', 
    #                 'coco/checkpoints/DualModel/dual_fuse_global3_rl_last.pth', 
    #                 'coco/checkpoints/DualModel/dual_fuse_global5_rl_best.pth']

    weight_files = ['coco/checkpoints/DualModel/dual_fuse_global_best.pth',
                    'coco/checkpoints/DualModel/dual_fuse_global2_best.pth', 
                    'coco/checkpoints/DualModel/dual_fuse_global3_last.pth', 
                    'coco/checkpoints/DualModel/dual_fuse_global5_best.pth']

    model = TransformerEnsemble(model, weight_files, args.device).to(args.device)

    # data = torch.load(args.model_path, map_location=args.device)
    # model.load_state_dict(data['state_dict_t'])
    # print(data['best_cider'])

    scores = predict_captions(model, dict_dataloader_val, text_field, args.device)
    print("Validation scores", scores)
    scores = predict_captions(model, dict_dataloader_test, text_field, args.device)
    print("Test scores", scores)