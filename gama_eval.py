""" This code is modified from https://github.com/val-iisc/GAMA-GAT """

import argparse
import os
from shutil import copyfile

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from utils import LinearModel

try:
    from torch.autograd.gradcheck import zero_gradients
except:
    import collections
    def zero_gradients(x):
        if isinstance(x, torch.Tensor):
            if x.grad is not None:
                x.grad.detach_()
                x.grad.zero_()
        elif isinstance(x, collections.abc.Iterable):
            for elem in x:
                zero_gradients(elem)
import wandb
from torchvision import datasets, transforms


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--arch', type=str, default='resnet18')

    # data
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--root', default='../../datasets/cifar10/val/', type=str)

    # wandb
    parser.add_argument('--project', type=str, required=True)
    parser.add_argument('--entity', type=str, required=True)
    parser.add_argument('--run_name', type=str, default='')
    parser.add_argument('--id', type=str, default='')
    parser.add_argument('--offline', action='store_true')
    return parser

def FGSM_Attack_step(model,loss,img,target,eps=0.,steps=30): 
    eps = eps/steps 
    for step in range(steps):
        img = Variable(img,requires_grad=True)
        zero_gradients(img) 
        out  = model(img)
        cost = loss(out,target)
        cost.backward()
        # print(type(img.grad))
        per = eps * torch.sign(img.grad.data)
        adv = img.data + per
        img = torch.clamp(adv, 0., 1.)
    return img.detach()

def max_margin_logit_loss(logits,y, num_classes=10):
    # Logit score of correct class
    logit_org = logits.gather(1,y.view(-1,1))
    # Second largest logit score
    logit_target = logits.gather(1,(logits - torch.eye(num_classes)[y].to("cuda") * 9999).argmax(1, keepdim=True))
    loss = -logit_org + logit_target
    loss = torch.mean(loss)
    return loss
    
def MSPGD_MT_Bern(model,loss,data,target,num_classes,eps=0.1,eps_iter=0.1,bounds=[],steps=[7,20,50,100],w_reg=25,lin=50,SCHED=[],drop=1):
    """
    model
    loss : loss used for training
    data : input to network
    target : ground truth label corresponding to data
    eps : perturbation srength added to image
    eps_iter
    """
    #Raise error if in training mode
    if model.training:
        assert 'Model is in  training mode'
    tar = Variable(target.cuda())
    data = data.cuda()
    B,C,H,W = data.size()
    noise  = torch.FloatTensor(np.random.uniform(-eps,eps,(B,C,H,W))).cuda()
    noise  = eps*torch.sign(noise)
    img_arr = []
    W_REG = w_reg
    orig_img = data+noise
    orig_img = Variable(orig_img,requires_grad=True)
    for step in range(steps[-1]):
        # convert data and corresponding into cuda variable
        img = data + noise
        img = Variable(img,requires_grad=True)
        
        if step in SCHED:
            eps_iter /= drop
        
        # make gradient of img to zeros
        zero_gradients(img) 
        # forward pass        
        orig_out = model(orig_img)
        P_out = nn.Softmax(dim=1)(orig_out)
        
        out  = model(img)
        Q_out = nn.Softmax(dim=1)(out)
        #compute loss using true label
        if step <= lin:
            cost =  W_REG*((P_out - Q_out)**2.0).sum(1).mean(0) + max_margin_logit_loss(Q_out,tar, num_classes)
            W_REG -= w_reg/lin
        else:
            cost = max_margin_logit_loss(Q_out,tar, num_classes)
        #backward pass
        cost.backward()
        #get gradient of loss wrt data
        per =  torch.sign(img.grad.data)
        #convert eps 0-1 range to per channel range 
        per[:,0,:,:] = (eps_iter * (bounds[0,1] - bounds[0,0])) * per[:,0,:,:]
        if(per.size(1)>1):
            per[:,1,:,:] = (eps_iter * (bounds[1,1] - bounds[1,0])) * per[:,1,:,:]
            per[:,2,:,:] = (eps_iter * (bounds[2,1] - bounds[2,0])) * per[:,2,:,:]
        #  ascent
        adv = img.data + per.cuda()
        #clip per channel data out of the range
        img.requires_grad =False
        img[:,0,:,:] = torch.clamp(adv[:,0,:,:],bounds[0,0],bounds[0,1])
        if(per.size(1)>1):
            img[:,1,:,:] = torch.clamp(adv[:,1,:,:],bounds[1,0],bounds[1,1])
            img[:,2,:,:] = torch.clamp(adv[:,2,:,:],bounds[2,0],bounds[2,1])
        img = img.data
        noise = img - data
        noise  = torch.clamp(noise,-eps,eps)

        for j in range(len(steps)):
            if step == steps[j]-1:
                img_tmp = data + noise
                img_arr.append(img_tmp)
                break
    return img_arr


def main(args):

    ###################################### Logging ##################################################
    if args.dataset == 'cifar10':
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        args.num_classes = 100
    else:
        raise ValueError
    results_dir = os.path.dirname(args.ckpt)
    EVAL_LOG_NAME = os.path.join(results_dir, f'gama_eval.txt')

    ###################################### Load checkpoint ##################################################
    log_file = open(EVAL_LOG_NAME,'a+')
    model = LinearModel(args.num_classes, args)
    model = nn.DataParallel(model)
    model.cuda()
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict, strict=True) 
    load_msg = f'Loaded checkpoint: {args.ckpt}'
    print(load_msg)
    log_file.write(load_msg)
    model.eval()

    ###################################### Load dataset ##################################################
    transform_test = transforms.Compose([transforms.ToTensor()])
    data_cls = datasets.CIFAR100 if args.dataset == 'cifar100' else datasets.CIFAR10
    test_set   = data_cls(root=args.root, train=False, download=True, transform=transform_test)
    BATCH_SIZE = 100
    test_size  = len(test_set)
    test_loader   = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE)
    dset_msg = f'{args.dataset} loading done.\n'
    print(dset_msg)
    log_file.write(dset_msg)
    log_file.close()

    ###################################### Eval ##################################################
    loss = nn.CrossEntropyLoss()

    ##################################### FGSM #############################################
    log_file = open(EVAL_LOG_NAME,'a+')
    fgsm_steps = 1
    hash_tag = '#'*20
    msg = f'{hash_tag} FGSM (steps={fgsm_steps}) {hash_tag}\n'
    log_file.write(msg)
    print(msg)
    for eps in np.arange(0.0/255, 10.0/255, 2.0/255):
        accuracy = 0
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            data = FGSM_Attack_step(model,loss,data,target,eps=eps,steps=fgsm_steps)
            with torch.no_grad():
                out = model(data)
            prediction = out.data.max(1)[1] 
            accuracy += prediction.eq(target.data).sum().item()
        acc = (accuracy * 100) / test_size
        msg = f'eps {eps}, Acc {acc}\n'
        print(msg)
        log_file.write(msg)

    log_file.close()

    ##################################### iFGSM #############################################
    log_file = open(EVAL_LOG_NAME,'a+')
    ifgsm_steps = 7
    hash_tag = '#'*20
    msg = f'{hash_tag} iFGSM: step={ifgsm_steps} {hash_tag}\n'
    log_file.write(msg)
    print(msg)

    for eps in np.arange(2.0/255, 10.0/255, 2.0/255):
        accuracy = 0
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            data = FGSM_Attack_step(model,loss,data,target,eps=eps,steps=ifgsm_steps)
            with torch.no_grad():
                out = model(data)
            prediction = out.data.max(1)[1] 
            accuracy = accuracy + prediction.eq(target.data).sum()
        acc = (accuracy.item()*1.0) / (test_size) * 100
        msg = f'eps {eps}, Acc {acc}\n'
        print(msg)
        log_file.write(msg)
        
    log_file.close()

    ##################################### PGD, steps=[7,20,50,100,500] #############################################
    log_file = open(EVAL_LOG_NAME,'a+')
    SCHED = [60,85]
    drop = 10    
    lin = 25
    w_reg = 50
    all_steps = [60,85,90,100]
    eps_iter = 16
    hash_tag = '#'*20
    msg = f'{hash_tag} Gama-PGD Wreg{w_reg} lin{lin}, drop by {drop} at {SCHED}: steps={all_steps}, eps_iter_init={eps_iter}/255 {hash_tag}\n'
    print(msg)
    log_file.write(msg)
    num_steps = len(all_steps)
    eps_iter /= 255
    eps = 8.0/255
    acc_arr = torch.zeros((num_steps))
    for data, target in test_loader:
        adv_arr = MSPGD_MT_Bern(model,loss,data,target,args.num_classes,eps=eps,eps_iter=eps_iter,bounds=np.array([[0,1],[0,1],[0,1]]),steps=all_steps,w_reg=w_reg,lin=lin,SCHED=SCHED,drop=drop)
        target = Variable(target).cuda()
        for j in range(num_steps):
            data   = Variable(adv_arr[j]).cuda()
            with torch.no_grad():
                out = model(data)
            prediction = out.data.max(1)[1] 
            acc_arr[j] = acc_arr[j] + prediction.eq(target.data).sum()

    for j in range(num_steps):
        acc_arr[j] = (acc_arr[j].item()*1.0) / (test_size) * 100
        msg = msg = f'eps {eps}, steps {all_steps[j]}, Acc {acc_arr[j]}\n'
        print(msg)
        log_file.write(msg)

    log_file.close()
    copyfile(EVAL_LOG_NAME, os.path.join(wandb.run.dir, 'gama_eval.txt'))
    return acc_arr[-1]


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    if not args.id:
        args.id = args.ckpt.split('/')[1]
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
