import argparse
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import wandb
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from tqdm import tqdm

from utils import (LinearModel, Logger, evaluate_adv, fix_bn, fix_model,
                   setup_seed)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--experiment', type=str, help='exp name', default='')

    # data
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--test_batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--start_epoch', default=0, type=int)

    # model
    parser.add_argument('--arch', type=str, default='WideResNet34')
    parser.add_argument('--trainmode', default='adv', type=str, help='adv or normal or test')
    parser.add_argument('--fixmode', default='f1', type=str, help='f1: fix nothing, f2: fix previous 3 stages, f3: fix all except fc')
    parser.add_argument('--fixbn', action='store_true', help='if specified, fix bn for the layers been fixed')

    # attack details
    parser.add_argument('--epsilon', type=float, default=8. / 255.)
    parser.add_argument('--num_steps_train', type=int, default=10)
    parser.add_argument('--num_steps_test', type=int, default=20)
    parser.add_argument('--step_size', type=float, default=2. / 255.)
    parser.add_argument('--beta', type=float, default=6.0, help='regularization, i.e., 1/lambda in TRADES')

    # lr, optimizer and scheduler
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', '--wd', default=2e-4, type=float)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--decreasing_lr', default='15,20', help='multistep LR decay milestones')

    # logging and checkpoint
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_freq', '-s', default=1, type=int)
    parser.add_argument('--checkpoint', default='', type=str)
    
    # wandb
    parser.add_argument('--project', type=str, required=True)
    parser.add_argument('--entity', type=str, required=True)
    parser.add_argument('--id', default=wandb.util.generate_id(), help='wandb id to resume run')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--name', default='', help='wandb run name')

    return parser

def get_loaders(args):
    T_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    T_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    if args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(
            root=args.root, train=True, download=True, transform=T_train)
        valset = torchvision.datasets.CIFAR10(
            root=args.root, train=True, download=True, transform=T_test)
        testset = torchvision.datasets.CIFAR10(
            root=args.root, train=False, download=True, transform=T_test)
        # create class balanced val-set
        train_indices = list(range(50000))
        val_indices = []
        count = np.zeros(10)
        for index in range(len(trainset)):
            _, target = trainset[index]
            if np.all(count==100):
                break
            if count[target] < 100:
                count[target] += 1
                val_indices.append(index)
                train_indices.remove(index)
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
    
    elif args.dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(
            root=args.root, train=True, download=True, transform=T_train)
        valset = torchvision.datasets.CIFAR100(
            root=args.root, train=True, download=True, transform=T_test)
        testset = torchvision.datasets.CIFAR100(
            root=args.root, train=False, download=True, transform=T_test)
        # create class balanced val-set
        train_indices = list(range(50000))
        val_indices = []
        count = np.zeros(100)
        for index in range(len(trainset)):
            _, target = trainset[index]
            if np.all(count==10):
                break
            if count[target] < 10:
                count[target] += 1
                val_indices.append(index)
                train_indices.remove(index)
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

    kwargs = {'num_workers': 16}
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)
    vali_loader = torch.utils.data.DataLoader(
        valset, batch_size=args.batch_size, sampler=valid_sampler, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    return train_loader, vali_loader, test_loader

def compute_loss(model, x, y, optimizer, step_size, epsilon, perturb_steps, beta, trainmode, fixbn, fixmode):
    if trainmode == "adv":
        batch_size = len(x)
        criterion_kl = nn.KLDivLoss(reduction='sum').cuda()
        model.eval()
        x_adv = x.detach() + 0.001 * torch.randn(x.shape).cuda(x.device).detach()
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1), F.softmax(model(x), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(
                torch.max(x_adv, x - epsilon), x + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    optimizer.zero_grad()
    model.train()

    if fixbn:
        fix_bn(model, fixmode)

    logits = model(x)
    loss = F.cross_entropy(logits, y)

    if trainmode == "adv":
        logits_adv = model(x_adv)
        loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits, dim=1))
        loss += beta * loss_robust
    return loss


def train(args, model, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader, leave=False)
    for batch_idx, (data, target) in enumerate(pbar):
        pbar.set_description_str(f"Epoch {epoch}")
        data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
        optimizer.zero_grad()
        loss = compute_loss(model,
                           x=data,
                           y=target,
                           optimizer=optimizer,
                           step_size=args.step_size,
                           epsilon=args.epsilon,
                           perturb_steps=args.num_steps_train,
                           beta=args.beta,
                           trainmode=args.trainmode,
                           fixbn=args.fixbn,
                           fixmode=args.fixmode)
        loss.backward()
        optimizer.step()
        pbar.set_postfix({"loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})
        if batch_idx % args.log_interval == 0:
            wandb.log({
                f'{args.wandb_panel_name}/epoch': epoch,
                f'{args.wandb_panel_name}/train loss': loss.item(),
                f'{args.wandb_panel_name}/lr': optimizer.param_groups[0]['lr'],
            })


def evaluate_clean(model, loader):
    model.eval()
    correct = 0
    whole = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            whole += len(target)
    test_accuracy = correct / whole
    return test_accuracy * 100


def main(args):
    setup_seed(args.seed)
    model_dir = os.path.join('checkpoints_pretrain', wandb.run.id)
    os.makedirs(model_dir, exist_ok=True)

    if args.trainmode == 'adv':
        args.wandb_panel_name = 'TRADES'
        print(f"Adv ckpt dump: {os.path.join(model_dir, 'ata_best_model.pt')}")
    elif args.trainmode == 'normal':
        args.wandb_panel_name = 'LINEAR'
        print(f"Clean ckpt dump: {os.path.join(model_dir, 'best_model.pt')}")
    else:
        args.wandb_panel_name = args.trainmode

    log = Logger(os.path.join(model_dir))
    log.info(f"run: {args.name}\n")

    train_loader, val_loader, test_loader = get_loaders(args)

    if args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_classes = 100
    else:
        raise ValueError
    model = LinearModel(num_classes, args)
    model = torch.nn.DataParallel(model)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)

    start_epoch = args.start_epoch

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint['model']
    status = model.load_state_dict(state_dict, strict=False)
    print(status)
    model.cuda()
    log.info('read checkpoint {}'.format(args.checkpoint))

    fix_model(model, args.fixmode)

    if args.trainmode == 'normal':
        best_ckpt_name = f'best_model_lr{args.lr}_{args.fixmode}_{args.fixbn}.pt'
    else:
        best_ckpt_name = f'ata_best_model_lr{args.lr}_beta{args.beta}_{args.fixmode}_{args.fixbn}.pt'
    best_acc = 0.
    for epoch in range(start_epoch + 1, args.epochs + 1):
        train(args, model, train_loader, optimizer, epoch)
        scheduler.step()
        if args.trainmode != 'normal':
            val_acc = evaluate_adv(model, val_loader, epsilon=args.epsilon, alpha=args.step_size,
                                    criterion=F.cross_entropy, log=log, attack_iter=args.num_steps_test)
        else:
            val_acc = evaluate_clean(model, val_loader)
        wandb.log({f'{args.wandb_panel_name}/val acc': val_acc})
        if val_acc > best_acc:
            print(f'Saving at epoch {epoch}')
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                # 'optim': optimizer.state_dict(),
                'best_acc': best_acc,
            }, os.path.join(model_dir, best_ckpt_name))

    # Evaluate on best model
    filename = os.path.join(model_dir, best_ckpt_name)
    best_ckpt = torch.load(filename)
    print(f"Evaluating checkpoint of epoch {best_ckpt['epoch']} (best)")
    model.load_state_dict(best_ckpt['state_dict'])
    test_tacc = evaluate_clean(model, test_loader)
    test_atacc = evaluate_adv(model, test_loader, epsilon=args.epsilon, alpha=args.step_size, criterion=F.cross_entropy, log=log, attack_iter=args.num_steps_test)
    log.info(f"On the {best_ckpt_name}, test tacc is {test_tacc}, test atacc is {test_atacc}")
    log_file = log.get_path()
    shutil.copyfile(log_file, os.path.join(wandb.run.dir, 'finetune_log.txt'))

    # gama evaluation
    torch.cuda.empty_cache()
    from gama_eval import get_parser as get_gama_parser
    from gama_eval import main as gama_eval
    parser = get_gama_parser()
    gama_args, _ = parser.parse_known_args()
    gama_args.ckpt = filename
    gama_args.name = args.name
    gama_args.dataset = args.dataset
    gama_args.arch = args.arch
    gama_args.root = args.root
    gama_acc = gama_eval(gama_args)
    return test_tacc, test_atacc, gama_acc


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    wandb.init(
        project=args.project, 
        entity=args.entity,
        id=args.id,
        name=args.name if args.name else None,
        resume=True,
        mode='offline' if args.offline else 'online',
        config=args,
        save_code=True,
    )
    main(args)
