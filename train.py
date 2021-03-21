import os
import sys
sys.path.insert(0, '../../')
import time
import glob
import numpy as np
import torch
import optimizers.darts.utils as utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import sota.cnn.genotypes as genotypes
from sota.cnn.spaces import spaces_dict

from torch.autograd import Variable
#from sota.cnn.model import Network
from sota.cnn.model_search_pcdarts import PCDARTSNetwork as Network
from torch.utils.tensorboard import SummaryWriter
from sota.cnn.grad_cam import GradCAM
import cv2
from sota.cnn.resnet import resnet18
from sota.cnn.dataset import ReweightDataset


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../../data',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='choose dataset')
parser.add_argument('--batch_size', type=int, default=128, help='batch size,96')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=1000, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--cutout_prob', type=float, default=1.0, help='cutout probability')
parser.add_argument('--cutmix_prob', type=float, default=0.5, help='cutmix probability')
parser.add_argument('--beta', type=float, default=0, help='beta sampling')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='exp', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='PixelDARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--search_space', type=str, default='s3', help='searching space to choose from')
args = parser.parse_args()

args.save = '../../experiments/sota/{}/eval-{}-{}-{}-{}'.format(
    args.dataset, args.save, time.strftime("%Y%m%d-%H%M%S"), args.arch, args.seed)
if args.cutout:
    args.save += '-cutout-' + str(args.cutout_length) + '-' + str(args.cutout_prob)
if args.auxiliary:
    args.save += '-auxiliary-' + str(args.auxiliary_weight)
args.save += '-' + str(np.random.randint(10000))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
writer = SummaryWriter(args.save + '/runs')


if args.dataset == 'cifar100':
    n_classes = 100
else:
    n_classes = 10


def main():
    torch.set_num_threads(3)
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    genotype = eval("genotypes.%s" % args.arch)
    #model = Network(args.init_channels, n_classes, args.layers, args.auxiliary, genotype)
    criterion = nn.CrossEntropyLoss().cuda()
    model = Network(args.init_channels, n_classes, args.layers, criterion, spaces_dict[args.search_space]).cuda()
    model = model.cuda()
    pretrained_dict = torch.load("weights-11.pt")
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    """
    if args.dataset == 'cifar10':
        train_transform, valid_transform = utils._data_transforms_cifar10(args)
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
    elif args.dataset == 'cifar100':
        train_transform, valid_transform = utils._data_transforms_cifar100(args)
        train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)
    """
    train_data = ReweightDataset("newdata", "target_train.csv")
    valid_data = ReweightDataset("newdata_valid", "target_valid.csv")

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs))

    
    best_acc_top1 = 0
    best_acc_top5 = 0
    for epoch in range(args.epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        if args.cutout:
            # increase the cutout probability linearly throughout search
            train_transform.transforms[-1].cutout_prob = args.cutout_prob * \
                epoch / (args.epochs - 1)
            logging.info('epoch %d lr %e cutout_prob %e', epoch, lr,
                         train_transform.transforms[-1].cutout_prob)
        else:
            logging.info('epoch %d lr %e', epoch, lr)
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        train_acc, train_acc_5, train_obj = train(train_queue, model, criterion, optimizer)
        logging.info('train_acc top 1 %f top 5 %f', train_acc, train_acc_5)
        writer.add_scalar('Acc/train', train_acc, epoch)
        writer.add_scalar('Obj/train', train_obj, epoch)
      
        valid_acc_top1, valid_acc_top5, valid_obj = infer(valid_queue, model, criterion)
#         logging.info('Valid_acc_top1: %f', valid_acc_top1)
        logging.info('valid_acc top 1 %f top 5 %f', valid_acc_top1, valid_acc_top5)
        writer.add_scalar('Acc/valid', valid_acc_top1, epoch)
        writer.add_scalar('Obj/valid', valid_obj, epoch)
        
        is_best = False
        if valid_acc_top5 > best_acc_top5:
            best_acc_top5 = valid_acc_top5
        if valid_acc_top1 > best_acc_top1:
            best_acc_top1 = valid_acc_top1
            is_best = True
        
        utils.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc_top1': best_acc_top1,
            'optimizer' : optimizer.state_dict(),
            }, is_best, args.save)

        utils.save(model, os.path.join(args.save, 'weights.pt'))      
    writer.close()

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def train(train_queue, model, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()
    #res = resnet18()
    #res = res.cuda()
    #res.load_state_dict(torch.load("resnet18.pt"))
    #gcam = GradCAM(model=res)
    #target_layer = "cells.9"
    #target_layer = "layer1"
    #target_csv = open("target_train.csv", "w")
    #target_csv.write("index,label\n")
    cnt = 0
    for step, (input, target) in enumerate(train_queue):
        """
        print (cnt)
        target_csv.write(str(cnt) + "," + str(target.item()) + "\n")
        if cnt == 12800:
            break
        input = input.cuda() #bxcxhxw
        ori_img = input[0, :, :, :].detach().cpu().numpy()
        target = target.cuda(non_blocking=True)
        new_img = np.zeros((33, 32, 32))
        new_img[:3, :, :] = ori_img
        for idx in range(10):
            _p = gcam.forward(input)
            gcam.backward(idx=idx)
            region = gcam.generate(target_layer=target_layer)
            cmap = cv2.resize(region, (32, 32))
            new_img[3*idx+3:3*idx+6, :, :] = cmap
        np.save("newdata/reweight_{}.npy".format(cnt), new_img)
        cnt += 1
        # mixup training
        alpha = 1
        use_cuda = True
        inputs, targets_a, targets_b, lam = mixup_data(inputs, target,
                                                       alpha, use_cuda)
        inputs, targets_a, targets_b = map(Variable, (inputs,
                                                      targets_a, targets_b))
        outputs = model(inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(outputs, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)
        """
        input = input.cuda() #bxcxhxw
        target = target.cuda(non_blocking=True)
        r = np.random.rand(1)
        if args.beta > 0 and r < args.cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(args.beta, args.beta)
            rand_index = torch.randperm(input.size()[0]).cuda()
            target_a = target
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            # compute output
            logits = model(input)
            loss = criterion(logits, target_a) * lam + criterion(logits, target_b) * (1. - lam)
        else:
            # compute output
            logits = model(input)
            loss = criterion(logits, target)
            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)

        optimizer.zero_grad()
        #logits = model(input)
        #loss = criterion(logits, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.report_freq == 0:
            logging.info('train %03d avg: %e top1: %f top5: %f', step, objs.avg, top1.avg, top5.avg)
            if 'debug' in args.save:
                break

    return top1.avg, top5.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda(non_blocking=True)

            logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
                if 'debug' in args.save:
                    break

    return top1.avg, top5.avg, objs.avg


if __name__ == '__main__':
    main()



