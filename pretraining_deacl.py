from __future__ import print_function

import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import wandb
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from autoaugment import CIFAR10Policy
from finetune import get_parser as get_finetune_parser
from finetune import main as finetune
from utils import (PretrainModel, adjust_learning_rate, save_checkpoint,
                   setup_seed, warmup_learning_rate)


def get_parser():
    parser = argparse.ArgumentParser()

    # experiment related
    parser.add_argument('--seed', default=0, type=float, help='random seed')
    parser.add_argument('-m', '--description', type=str, default="", help='details of the run')
    parser.add_argument('--epoch', type=int, default=1000, help='total epochs')
    parser.add_argument('--save_epoch', type=int, default=50, help='save epochs')

    # dataset and dataloader
    parser.add_argument('--dataset', type=str, default='cifar100', help='dataset')
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=512)

    # model
    parser.add_argument('--arch', type=str, default='WideResNet34')
    parser.add_argument('--proj_hdim', type=int, default=640)
    parser.add_argument('--proj_odim', type=int, default=256)
    parser.add_argument('--PT_ckpt', type=str, required=True)
    parser.add_argument('--student_scratch', action='store_true')

    # attack related
    parser.add_argument('--epsilon', type=float, default=8, help='The upper bound change of L-inf norm on input pixels')
    parser.add_argument('--iters', type=int, default=5, help='The number of iterations for iterative attacks')
    parser.add_argument('--step_size', type=int, default=2, help='Step size for iterative attacks')

    # loss related
    parser.add_argument('--beta', type=float, default=2)
    parser.add_argument('--w_proj', type=float, default=0.5)

    # LR, optimizers & schedulers
    parser.add_argument('--lr', type=float, default=0.5)
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--linear_lr', type=float, default=5.0)

    # wandb logging
    parser.add_argument('--project', type=str, required=True)
    parser.add_argument('--entity', type=str, required=True)
    parser.add_argument('--id', default=wandb.util.generate_id(), help='wandb id to resume run')
    parser.add_argument('--resume', action='store_true', help='Resume a previous wandb run')
    parser.add_argument('--offline', action='store_true')

    parser.add_argument('--linear_eval', action='store_true')

    return parser


def get_loader(args):
    PC_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    class MultiAugCIFAR10(datasets.CIFAR10):
        def __getitem__(self, index: int):
            img, target = self.data[index], self.targets[index]
            img = Image.fromarray(img)
            if isinstance(self.transform, list):
                out = [tx(img) for tx in self.transform]
                out += [target]
                return out
            return self.transform(img), target

    class MultiAugCIFAR100(datasets.CIFAR100):
        def __getitem__(self, index: int):
            img, target = self.data[index], self.targets[index]
            img = Image.fromarray(img)
            if isinstance(self.transform, list):
                out = [tx(img) for tx in self.transform]
                out += [target]
                return out
            return self.transform(img), target

    if args.dataset == 'cifar10': 
        DATASET_CLS = MultiAugCIFAR10
    elif args.dataset == 'cifar100': 
        DATASET_CLS = MultiAugCIFAR100
    else: 
        raise Exception('Unknown dataset')

    train_dataset = DATASET_CLS(root=args.root, transform=PC_transform)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8
    )
    return train_loader


class Attacker(nn.Module):
    def __init__(self, model, config):
        super(Attacker, self).__init__()
        self.model = model
        self.args = config['args']
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        self.epochs = config['epochs']

    def forward(self, x, target_feat):
        x_adv = x.detach().clone()
        eps = self.epsilon
        step_size = self.step_size

        self.model.eval()

        if self.rand:
            x_adv = x_adv + torch.zeros_like(x).uniform_(-eps, eps)

        for _ in range(self.num_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                feat = self.model(x_adv)
                loss = -F.cosine_similarity(feat, target_feat, dim=1).sum()
            grad_x_cl = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad_x_cl).detach().clone()
            x_adv = torch.min(torch.max(x_adv, x - eps), x + eps)
            x_adv = torch.clamp(x_adv, 0, 1)

        self.model.train()

        return x_adv


def train(attacker, student, clean_teacher, train_loader, optimizer, epoch, args):
    student.train()
    train_loss = 0
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(train_loader, leave=False, ncols=100)

    for batch_idx, (x, _) in enumerate(pbar):
        pbar.set_description_str(f"Epoch {epoch}")
        x = x.cuda(non_blocking=True)
        warmup_learning_rate(args, epoch + 1, batch_idx, len(train_loader), optimizer)

        student_clean_feats = student(x)
        teacher_clean_feats = clean_teacher(x)

        x_adv = attacker(x, teacher_clean_feats)
        student_adv_feats = student(x_adv)

        adv_loss = -F.cosine_similarity(student_adv_feats, student_clean_feats).mean()
        clean_loss = -F.cosine_similarity(teacher_clean_feats, student_clean_feats).mean()
        loss = clean_loss + args.beta * adv_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        train_loss += loss.item()

        pbar.set_postfix({"loss": loss.item()})

        # log metrics
        panel = 'pretraining'
        lr_dict = {}
        for i, grp in enumerate(optimizer.param_groups):
            lr_dict[f"{panel}/lr-{grp.get('name', f'grp{i}')}"] = grp['lr']
        metrics = {
            f'{panel}/clean loss': clean_loss,
            f'{panel}/adv loss': adv_loss,
            f'{panel}/total loss': loss,
        }
        wandb.log({**metrics, **lr_dict})


def init_pretrained(model, args):
    PT_ckpt = torch.load(args.PT_ckpt, map_location='cpu')
    status = model.load_state_dict(PT_ckpt['state_dict'], strict=False)
    print(status)
    return model


def main(args):
    setup_seed(args.seed)
    args.epochs = args.epoch
    args.decay = args.weight_decay
    args.warm = True
    args.warmup_from = 0.01
    args.warm_epochs = 10
    min_lr = args.lr * (args.lr_decay_rate ** 3)
    args.warmup_to = min_lr + (args.lr - min_lr) * (1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2

    train_loader = get_loader(args)
    student = PretrainModel(args)
    clean_teacher = PretrainModel(args)
    student.projector = nn.Identity()  # no projector
    clean_teacher.projector = nn.Identity()  # no projector
    
    # Load clean SSL pretrained model
    clean_teacher = init_pretrained(clean_teacher, args)
    if not args.student_scratch:
        student = init_pretrained(student, args)

    student = nn.DataParallel(student)
    student.cuda()
    clean_teacher = nn.DataParallel(clean_teacher)
    clean_teacher.cuda()
    for p in clean_teacher.parameters():
        p.requires_grad = False
    clean_teacher.eval()

    config = {
        'args': args,
        'epsilon': args.epsilon / 255.,
        'num_steps': args.iters,
        'step_size': args.step_size / 255,
        'random_start': True,
        'epochs': args.epochs
    }
    attacker = Attacker(student, config)
    trainable_params = student.parameters()
    optimizer = torch.optim.SGD(trainable_params, lr=args.lr, momentum=0.9, weight_decay=args.decay)

    for epoch in range(args.epoch + 1):
        adjust_learning_rate(args, optimizer, epoch + 1)
        train(attacker, student, clean_teacher, train_loader, optimizer, epoch, args)
        
        save_freq_check = epoch % args.save_epoch == 0
        last_ep_check = epoch == args.epoch
        if save_freq_check or last_ep_check:
            curr_ckpt_path = save_checkpoint(student, optimizer, epoch)

    if args.linear_eval:
        print('Starting Standard Linear Evaluation...')
        time.sleep(10)
        linear_eval(curr_ckpt_path, args)

def linear_eval(ckpt_path, args):
    finetune_parser = get_finetune_parser()
    linear_args, _ = finetune_parser.parse_known_args()
    linear_args.epochs = 25
    linear_args.batch_size = 512
    linear_args.test_batch_size = 256
    linear_args.decreasing_lr = '15,20'
    linear_args.weight_decay = 2e-4
    linear_args.momentum = 0.9
    linear_args.epsilon = 8/255
    linear_args.num_steps_train = 10
    linear_args.num_steps_test = 20
    linear_args.step_size = 2/255
    linear_args.fixmode = 'f3'
    linear_args.fixbn = True
    linear_args.start_epoch = 0
    linear_args.checkpoint = ckpt_path
    linear_args.id = wandb.run.id
    linear_args.name = exp_name
    linear_args.resume = False
    linear_args.dataset = args.dataset
    linear_args.data = args.root
    linear_args.arch = args.arch
    linear_args.trainmode = 'normal'
    linear_args.lr = args.linear_lr

    print(f'SLF: lr{linear_args.lr}, Fix mode {linear_args.fixmode}, FixBN {linear_args.fixbn}')
    clean, pgd20, gama = finetune(linear_args)
    linear_results = f'PT lr {args.lr}, lin lr {linear_args.lr}: {clean:.2f} {pgd20:.2f} {gama:.2f}\n'
    print(linear_results)

    with open(os.path.join('checkpoints_pretrain', wandb.run.id, 'SLF_results.txt'), 'a+') as f:
        f.write(linear_results)
    with open(os.path.join(wandb.run.dir, 'SLF_results.txt'), 'a+') as f:
        f.write(linear_results)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    args.description = f"_{args.description}" if args.description else ""
    exp_name = f'DeACL_baseline_{args.dataset}_PT{args.epoch}ep{args.description}'
    wandb.init(
        project=args.project, 
        entity=args.entity,
        id=args.id,
        name=exp_name,
        resume=args.resume,
        mode='offline' if args.offline else 'online',
        config=args,
        save_code=True,
    )
    main(args)
