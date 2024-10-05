""" This code uses the original AutoAttack codebase: https://github.com/fra31/auto-attack (MIT License) """

import argparse
import json
import os
import shutil

import torch
import torch.nn as nn
import wandb
from autoattack import AutoAttack
from torchvision import datasets, transforms

from utils import LinearModel, ModelwithLinear, get_model


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt', type=str)
    parser.add_argument('--arch', type=str, required=True)

    # data
    parser.add_argument('--dataset', default='cifar100', type=str)
    parser.add_argument('--root', required=True)

    # attack
    parser.add_argument('--epsilon', type=float, default=8)
    parser.add_argument('--batch_size', type=int, default=1024)

    # wandb
    parser.add_argument('--project', type=str, required=True)
    parser.add_argument('--entity', type=str, required=True)
    parser.add_argument('--run_name', type=str, default='')
    parser.add_argument('--id', type=str, default='')
    parser.add_argument('--offline', action='store_true')
    return parser


def main(args):
    args_dict = vars(args)
    args_str = json.dumps(args_dict)
    
    if args.dataset == 'cifar10':
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        args.num_classes = 100
    else:
        raise ValueError
    results_dir = os.path.dirname(args.ckpt)
    EVAL_LOG_NAME = os.path.join(results_dir, f'autoattack.txt')

    ###################################### Load checkpoint ##################################################
    log_file = open(EVAL_LOG_NAME,'a+')
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    try:
        model = LinearModel(args.num_classes, args)
        model = nn.DataParallel(model)
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict, strict=True) 
    except:
        print('DeACL model loading...')
        backbone, feat_dim = get_model(args)
        model = ModelwithLinear(backbone, feat_dim, args.num_classes)
        bb_status = model.model.load_state_dict(checkpoint['model'], strict=False)
        print(bb_status)
        model.classifier.load_state_dict({k.replace('classifier.', ''): v for k, v in checkpoint['classifier'].items()})
        model = nn.DataParallel(model)

    load_msg = f'Loaded checkpoint: {args.ckpt}'
    print(load_msg)
    log_file.write(load_msg)
    model.cuda()
    model.eval()

    transform_test = transforms.Compose([transforms.ToTensor()])
    data_cls = datasets.CIFAR100 if args.dataset == 'cifar100' else datasets.CIFAR10
    test_set   = data_cls(root=args.root, train=False, download=True, transform=transform_test)
    test_loader   = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size)
    dset_msg = f'{args.dataset} loading done.\n'
    print(dset_msg)
    log_file.write(dset_msg)
    log_file.close()

    l = [x for (x, _) in test_loader]
    x_test = torch.cat(l, 0)
    l = [y for (_, y) in test_loader]
    y_test = torch.cat(l, 0)

    # AA
    adversary = AutoAttack(model, norm='Linf', eps=args.epsilon / 255, version='standard', log_path=EVAL_LOG_NAME)
    adversary.logger.log(args_str)
    with torch.no_grad():
        adv_complete = adversary.run_standard_evaluation(x_test, y_test, bs=args.batch_size)

    shutil.copyfile(EVAL_LOG_NAME, os.path.join(wandb.run.dir, 'AA_log.txt'))


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    wandb.init(
        project=args.project, 
        entity=args.entity,
        id=args.id,
        name=args.run_name,
        resume=True,
        mode='offline' if args.offline else 'online',
        save_code=True,
    )
    main(args)
