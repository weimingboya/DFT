import random
import torch
import time
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import NLLLoss
import os
from shutil import copyfile
from torch.utils.tensorboard import SummaryWriter
from common.evaluation import PTBTokenizer, Cider
from common.data import DataLoader
from common.data.field import RawField
from common.utils import evaluate_loss, evaluate_metrics, train_xe, train_scst
from common.utils.utils import setup_seed

setup_seed(123456)

def train(args, model, datasets, image_field, text_field, optim=None, scheduler=None, 
          train_xe_fn = train_xe, evaluate_loss_fn = evaluate_loss):
          
    device = args.device
    output = args.output
    use_rl = args.use_rl

    date = time.strftime("%Y-%m-%d", time.localtime())
    writer = SummaryWriter(log_dir=os.path.join(output, 'tensorboard_logs', args.exp_name, date))

    train_dataset, val_dataset, test_dataset = datasets

    dict_dataset_val = val_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})

    cider_cache_path = 'cache/cider_cache.pkl'
    if use_rl:
        if os.path.exists(cider_cache_path):
            cider_train = torch.load(cider_cache_path)
        else:
            ref_caps_train = list(train_dataset.text)
            cider_train = Cider(PTBTokenizer.tokenize(ref_caps_train))            
            torch.save(cider_train, cider_cache_path)

        train_dataset = train_dataset.image_dictionary({'image': image_field, 'text': RawField()})


    train_batch_size = args.batch_size // 5 if use_rl else args.batch_size
    dataloader_train = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=False)
    dataloader_val = DataLoader(val_dataset, batch_size=train_batch_size, num_workers=args.workers, pin_memory=True, drop_last=False)
    dict_dataloader_val = DataLoader(dict_dataset_val, batch_size=train_batch_size, num_workers=args.workers, pin_memory=True, drop_last=False)
    dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=train_batch_size, num_workers=args.workers, pin_memory=True, drop_last=False)

    # def lambda_lr(s):
    #     warm_up = args.warmup
    #     s += 1
    #     return (model.d_model ** -.5) * min(s ** -.5, s * warm_up ** -1.5)

    def lambda_lr(s):
        if s <= 3:
            lr = args.xe_base_lr * s / 4
        elif s <= 10:
            lr = args.xe_base_lr
        elif s <= 12:
            lr = args.xe_base_lr * 0.2
        else:
            lr = args.xe_base_lr * 0.2 * 0.2
        return lr
    
    def lambda_lr_rl(s):
        refine_epoch = 8
        if s <= refine_epoch:
            lr = args.rl_base_lr
        elif s <= refine_epoch + 3:
            lr = args.rl_base_lr * 0.2
        elif s <= refine_epoch + 6:
            lr = args.rl_base_lr * 0.2 * 0.2
        else:
            lr = args.rl_base_lr * 0.2 * 0.2 * 0.2
        return lr

    # Initial conditions
    if use_rl:
        optim = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
        scheduler = LambdaLR(optim, lambda_lr_rl)
    else:
        optim = Adam(model.parameters(), lr=1, betas=(0.9, 0.98)) if optim is None else optim
        scheduler = LambdaLR(optim, lambda_lr) if scheduler is None else scheduler
    

    loss_fn = NLLLoss(ignore_index=text_field.vocab.stoi['<pad>'])
    best_cider = .0
    patience = 0
    start_epoch = 0

    last_saved = os.path.join('coco/checkpoints', output, '%s_last.pth' % args.exp_name)
    best_saved = os.path.join('coco/checkpoints', output, '%s_best.pth' % args.exp_name)

    if args.resume_last or args.resume_best:
        if use_rl:
            last_saved = os.path.join('coco/checkpoints', output, '%s_rl_last.pth' % args.exp_name)
            best_saved = os.path.join('coco/checkpoints', output, '%s_rl_best.pth' % args.exp_name)

        if args.resume_last:
            fname = last_saved
        else:
            fname = best_saved

        if os.path.exists(fname):
            data = torch.load(fname)
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'], strict=False)
            optim.load_state_dict(data['optimizer'])
            scheduler.load_state_dict(data['scheduler'])
            start_epoch = data['epoch'] + 1
            best_cider = data['best_cider']
            patience = data['patience']
            print('Resuming from epoch %d, validation loss %f, and best cider %f' % (
                data['epoch'], data['val_loss'], data['best_cider']))

    elif use_rl:
        data = torch.load(best_saved, map_location=device)
        model.load_state_dict(data['state_dict'], strict=False)
        best_cider = data['best_cider']
        start_epoch = 0
        patience = 0
        print('Resuming from XE epoch %d, validation loss %f, and best cider %f' % (
            data['epoch'], data['val_loss'], data['best_cider']))

        last_saved = os.path.join('coco/checkpoints', output, '%s_rl_last.pth' % args.exp_name)
        best_saved = os.path.join('coco/checkpoints', output, '%s_rl_best.pth' % args.exp_name)

    print("Training starts")
    for e in range(start_epoch, start_epoch + 100):
        if not use_rl:
            train_loss = train_xe_fn(model, dataloader_train, optim, loss_fn, text_field, e, device, scheduler, args)
            writer.add_scalar('data/train_loss', train_loss, e)
        else:
            train_loss, reward, reward_baseline = train_scst(model, dataloader_train, optim, 
                                                             cider_train, text_field, e, device, scheduler, args)
            writer.add_scalar('data/train_loss', train_loss, e)
            writer.add_scalar('data/reward', reward, e)
            writer.add_scalar('data/reward_baseline', reward_baseline, e)

        # Validation loss
        val_loss = evaluate_loss_fn(model, dataloader_val, loss_fn, text_field, e, device, args)
        writer.add_scalar('data/val_loss', val_loss, e)

        # Validation scores
        scores = evaluate_metrics(model, dict_dataloader_val, text_field, e, device, args)
        print("Validation scores", scores)
        val_cider = scores['CIDEr']
        writer.add_scalar('data/val_cider', val_cider, e)
        writer.add_scalar('data/val_bleu1', scores['BLEU'][0], e)
        writer.add_scalar('data/val_bleu4', scores['BLEU'][3], e)
        writer.add_scalar('data/val_meteor', scores['METEOR'], e)
        writer.add_scalar('data/val_rouge', scores['ROUGE'], e)

        # Test scores
        scores = evaluate_metrics(model, dict_dataloader_test, text_field, e, device, args)
        print("Test scores", scores)
        writer.add_scalar('data/test_cider', scores['CIDEr'], e)
        writer.add_scalar('data/test_bleu1', scores['BLEU'][0], e)
        writer.add_scalar('data/test_bleu4', scores['BLEU'][3], e)
        writer.add_scalar('data/test_meteor', scores['METEOR'], e)
        writer.add_scalar('data/test_rouge', scores['ROUGE'], e)

        # Prepare for next epoch
        best = False
        if val_cider >= best_cider:
            best_cider = val_cider
            patience = 0
            best = True
        else:
            patience += 1

        # switch_to_rl = False
        exit_train = False
        # automatic training strategy 
        if patience == 5:
            if e < 15:
                patience = 0
            else:
                print('patience reached.')
                exit_train = True

        saved_dir = os.path.join('coco/checkpoints', output)
        if not os.path.exists(saved_dir):
            os.makedirs(saved_dir)
        
        torch.save({
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate(),
            'epoch': e,
            'val_loss': val_loss,
            'val_cider': val_cider,
            'state_dict': model.state_dict(),
            'optimizer': optim.state_dict(),
            'scheduler': scheduler.state_dict(),
            'patience': patience,
            'best_cider': best_cider,
            'use_rl': use_rl,
        }, last_saved)

        if best:
            copyfile(last_saved, best_saved)

        if exit_train:
            writer.close()
            break