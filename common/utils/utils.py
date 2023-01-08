import random
import torch
import os, pickle
import numpy as np
from tqdm import tqdm
import requests
import itertools
import multiprocessing
from common.data.dataset import COCODataset
import common.evaluation as evaluation
from torch import Tensor

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
    #  torch.backends.cudnn.deterministic = True

def one_hot_to_index(one_hot: Tensor) -> Tensor:
    """
    Converts a one-hot tensor into a tensor with corresponding indexes
    """
    device, dtype = one_hot.device, one_hot.dtype
    vocab_size = one_hot.shape[-1]
    oh2idx = torch.tensor(range(vocab_size), dtype=dtype, device=device)
    return (one_hot @ oh2idx.unsqueeze(dim=1)).long().squeeze(dim=-1)

def download_from_url(url, path):
    """Download file, with logic (from tensor2tensor) for Google Drive"""
    if 'drive.google.com' not in url:
        print('Downloading %s; may take a few minutes' % url)
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        with open(path, "wb") as file:
            file.write(r.content)
        return
    print('Downloading from Google Drive; may take a few minutes')
    confirm_token = None
    session = requests.Session()
    response = session.get(url, stream=True)
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            confirm_token = v

    if confirm_token:
        url = url + "&confirm=" + confirm_token
        response = session.get(url, stream=True)

    chunk_size = 16 * 1024
    with open(path, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)


def create_dataset(args, image_field, text_field):
    # Create the dataset
    dataset = COCODataset(image_field, text_field, args.image_folder, args.annotation_folder, args.annotation_folder)
    train_dataset, val_dataset, test_dataset = dataset.splits

    vocab_path = 'cache/vocab.pkl'
    if not os.path.isfile(vocab_path):
        print("Building vocabulary")
        text_field.build_vocab(train_dataset, val_dataset, min_freq=5)
        pickle.dump(text_field.vocab, open(vocab_path, 'wb'))
    else:
        text_field.vocab = pickle.load(open(vocab_path, 'rb'))

    return (train_dataset, val_dataset, test_dataset)


def evaluate_loss(model, dataloader, loss_fn, text_field, epoch, device = 'cuda', args=None):
    # Validation loss
    model.eval()
    running_loss = .0
    with tqdm(desc='Epoch %d - validation' % epoch, unit='it', total=len(dataloader)) as pbar:
        with torch.no_grad():
            for it, (images, captions) in enumerate(dataloader):
                # if it == 10:
                #     break
                captions = captions.to(device)
                if isinstance(images, tuple) or isinstance(images, list):
                    images = [x.to(device) for x in images]
                else:
                    images = images.to(device)
                out = model(images, captions)
                captions = captions[:, 1:].contiguous()
                out = out[:, :-1].contiguous()
                loss = loss_fn(out.view(-1, len(text_field.vocab)), captions.view(-1))
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()

    val_loss = running_loss / len(dataloader)
    return val_loss

def dict_to_cuda(input_dict, deivce):
    for key in input_dict:
        if isinstance(input_dict[key], list):
            input_dict[key] = [ val.to(deivce) for val in input_dict[key]]
        elif isinstance(input_dict[key], dict):
            dict_to_cuda(input_dict[key], deivce)
        else:
            input_dict[key] = input_dict[key].to(deivce)

def evaluate_metrics(model, dataloader, text_field, epoch, device = 'cuda', args=None):
    import itertools
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='Epoch %d - evaluation' % epoch, unit='it', total=len(dataloader)) as pbar:
        for it, (images, caps_gt) in enumerate(iter(dataloader)):
            # if it == 10:
            #     break
            with torch.no_grad():
                if isinstance(images, tuple) or isinstance(images, list):
                    images = [x.to(device) for x in images]
                else:
                    images = images.to(device)
                # images[0] = images[0].to(device)
                # dict_to_cuda(images[1], device)
                # images[0] = images[0].to(device)
                # images[1] = images[1].to(device)
                # images[2] = {
                #         k1: {
                #             k2: v2.to(device)
                #             for k2, v2 in v1.items()
                #         }
                #         for k1, v1 in images[2].items()
                #     }
                
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


def train_xe(model, dataloader, optim, loss_fn, text_field, epoch, device = 'cuda', scheduler = None, args=None):
    # Training with cross-entropy
    model.train()
    if scheduler is not None:
        scheduler.step()
    # print('lr0 = ', optim.state_dict()['param_groups'][0]['lr'])
    # print('lr1 = ', optim.state_dict()['param_groups'][1]['lr'])
    running_loss = .0
    with tqdm(desc='Epoch %d - train' % epoch, unit='it', total=len(dataloader)) as pbar:
        for it, (images, captions) in enumerate(dataloader):
            # if it == 10:
            #     break
            captions = captions.to(device)
            if isinstance(images, tuple) or isinstance(images, list):
                images = [x.to(device) for x in images]
            else:
                images = images.to(device)
            out = model(images, captions)
            optim.zero_grad()
            captions_gt = captions[:, 1:].contiguous()
            out = out[:, :-1].contiguous()
            loss = loss_fn(out.view(-1, out.shape[-1]), captions_gt.view(-1))
            loss.backward()

            optim.step()
            this_loss = loss.item()
            running_loss += this_loss

            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()
            # if scheduler is not None:
            #     scheduler.step()

    loss = running_loss / len(dataloader)
    return loss


def train_scst(model, dataloader, optim, cider, text_field, epoch, device = 'cuda', scheduler = None, args=None):
    # Training with self-critical
    model.train()
    if scheduler is not None:
        scheduler.step()
    lr = optim.state_dict()['param_groups'][0]['lr']

    tokenizer_pool = multiprocessing.Pool()
    running_reward = .0
    running_reward_baseline = .0
    running_loss = .0
    seq_len = 20
    beam_size = 5

    with tqdm(desc='Epoch %d - train' % epoch, unit='it', total=len(dataloader)) as pbar:
        for it, (images, caps_gt) in enumerate(dataloader):
            # if it == 2:
            #     break
            if isinstance(images, tuple) or isinstance(images, list):
                images = [x.to(device) for x in images]
                bs = images[0].shape[0]
            else:
                images = images.to(device)
                bs = images.shape[0]
            outs, log_probs = model.beam_search(images, seq_len, text_field.vocab.stoi['<eos>'],
                                                beam_size, out_size=beam_size)
            optim.zero_grad()

            # Rewards
            caps_gen = text_field.decode(outs.view(-1, seq_len))
            caps_gt = list(itertools.chain(*([c, ] * beam_size for c in caps_gt)))
            caps_gen, caps_gt = tokenizer_pool.map(evaluation.PTBTokenizer.tokenize, [caps_gen, caps_gt])
            reward = cider.compute_score(caps_gt, caps_gen)[1].astype(np.float32)
            reward = torch.from_numpy(reward).to(device).view(bs, beam_size)
            reward_baseline = torch.mean(reward, -1, keepdim=True)
            loss = -torch.mean(log_probs, -1) * (reward - reward_baseline)

            loss = loss.mean()
            loss.backward()
            optim.step()

            running_loss += loss.item()
            running_reward += reward.mean().item()
            running_reward_baseline += reward_baseline.mean().item()
            pbar.set_postfix(loss=running_loss / (it + 1), reward=running_reward / (it + 1), lr=lr)
            pbar.update()

    loss = running_loss / len(dataloader)
    reward = running_reward / len(dataloader)
    reward_baseline = running_reward_baseline / len(dataloader)
    return loss, reward, reward_baseline