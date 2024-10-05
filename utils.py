import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

import models.wideresnet as wideresnets
import wandb
from models.resnet import resnet18


class PretrainModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone, self.features_dim = get_model(args)
        if args.dataset == 'cifar10': num_classes = 10
        elif args.dataset == 'cifar100': num_classes = 100
        else: raise ValueError
        self.classifier = nn.Linear(self.features_dim, num_classes)
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, args.proj_hdim),
            nn.ReLU(),
            nn.Linear(args.proj_hdim, args.proj_odim),
        )

    def forward(self, x, proj=False, return_feat=False, linear=False):
        x = self.backbone(x)
        if linear and hasattr(self, 'classifier'): 
            return self.classifier(x), x
        if proj: 
            if return_feat:
                return self.projector(x), x
            return self.projector(x)
        return x


class LinearModel(nn.Module):
    def __init__(self, num_classes, args):
        super().__init__()
        self.backbone, self.features_dim = get_model(args)
        self.fc = nn.Linear(self.features_dim, num_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x


class ModelwithLinear(nn.Module):
    def __init__(self, model, inplanes, num_classes=10):
        super(ModelwithLinear, self).__init__()
        self.model = model
        
        self.classifier = nn.Linear(inplanes, num_classes)

    def forward(self, img):
        x = self.model(img)
        out = self.classifier(x)
        return out


class Logger(object):
    def __init__(self, path):
        self.path = path
        self.file = os.path.join(self.path, 'log.txt')

    def info(self, msg):
        print(msg)
        with open(self.file, 'a') as f:
            f.write(msg + "\n")

    def get_path(self):
        return self.file


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

def get_model(args):
    if 'resnet' in args.arch:
        print('Initializing resnet backbone...')
        backbone = resnet18()
        features_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        if 'cifar' in args.dataset:
            backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
            backbone.maxpool = nn.Identity()
    elif 'wideresnet' in args.arch.lower():
        print('Initializing wideresnet backbone...')
        backbone = getattr(wideresnets, args.arch)()
        features_dim = backbone.nChannels
        backbone.fc = nn.Identity()
    else:
        raise ValueError
    return backbone, features_dim

def fix_bn(model, fixmode):
    if fixmode == 'f1':
        # fix none
        pass
    elif fixmode == 'f2':
        # fix previous three layers
        for name, m in model.named_modules():
            if not ("layer4" in name or "fc" in name):
                m.eval()
    elif fixmode == 'f3':
        # fix every layer except fc
        # fix previous four layers
        for name, m in model.named_modules():
            if not ("fc" in name or 'classifier' in name):
                m.eval()
    else:
        assert False

def fix_model(model, fixmode):
    if fixmode == 'f1':
        # fix none
        pass
    elif fixmode == 'f2':
        # fix previous three layers
        for name, param in model.named_parameters():
            if not ("layer4" in name or "fc" in name):
                param.requires_grad = False
            else:
                print("trainable {}".format(name))
    elif fixmode == 'f3':
        # fix every layer except fc; fix previous four layers
        for name, param in model.named_parameters():
            if not ("fc" in name or 'classifier' in name):
                param.requires_grad = False
            else:
                print("trainable {}".format(name))
    else:
        assert False

@torch.enable_grad()
def pgd_attack(model, images, labels, eps=8. / 255., alpha=2. / 255., iters=20, advFlag=None, forceEval=True, randomInit=True):
    loss = nn.CrossEntropyLoss().cuda()

    if randomInit:
        delta = torch.rand_like(images) * eps * 2 - eps
    else:
        delta = torch.zeros_like(images)
    delta = torch.nn.Parameter(delta, requires_grad=True)

    model.eval()
    for _ in range(iters):
        model.zero_grad()
        outputs = model(images + delta)
        cost = loss(outputs, labels)
        delta_grad = torch.autograd.grad(cost, [delta])[0]
        delta.data = delta.data + alpha * delta_grad.sign()
        delta.grad = None
        delta.data = torch.clamp(delta.data, min=-eps, max=eps)
        delta.data = torch.clamp(images + delta.data, min=0, max=1) - images

    model.zero_grad()
    return (images + delta).detach()

def evaluate_adv(model, test_loader, epsilon, alpha, criterion, log, attack_iter=40):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # fix random seed for testing
    torch.manual_seed(1)

    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        input, target = input.cuda(non_blocking=True), target.cuda(non_blocking=True)
        input_adv = pgd_attack(model, input, target, eps=epsilon, iters=attack_iter, alpha=alpha).data

        # compute output
        output = model.eval()(input_adv)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, = accuracy(output.data, target, topk=(1,))
        top1.update(prec1, input.size(0))
        losses.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
    return top1.avg.item()

def save_checkpoint(model, optimizer, epoch):
    print('=====> Saving checkpoint...')
    save_dir = f'./checkpoints_pretrain/{wandb.run.id}'
    os.makedirs(save_dir, exist_ok=True)
    state = {
            'model': model.state_dict(),
            'optim': optimizer.state_dict(),
            'epoch': epoch,
            'rng_state': torch.get_rng_state()
        }
    filename = f"{save_dir}/epoch_{epoch}.ckpt"
    torch.save(state, filename)
    return filename

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr
    eta_min = lr * (args.lr_decay_rate ** 3)
    lr = eta_min + (lr - eta_min) * (
            1 + math.cos(math.pi * epoch / args.epochs)) / 2

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
