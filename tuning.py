
import os
import numpy as np
import time, datetime
import argparse
import copy
from thop import profile
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed

import torchvision
from torchvision import datasets, transforms

from models.vgg import vgg_16_bn
from models.resnet import resnet_56, resnet_110

from torch.utils.data import DataLoader

from data import cifar10
import utils.common as utils


parser = argparse.ArgumentParser("training")

parser.add_argument(
    '--data_dir',
    type=str,
    default='./data/',
    help='path to dataset')

parser.add_argument(
    '--arch',
    type=str,
    choices=('vgg_16_bn','resnet_56','resnet_110'),
    help='architecture')

parser.add_argument(
    '--dataset',
    type=str,
    default='cifar10',
    choices=('cifar10', 'cifar100', 'ISIC', 'IDRID'),
    help='dataset')

parser.add_argument(
    '--job_dir',
    type=str,
    default='./models',
    help='path for saving trained models')

parser.add_argument(
    '--batch_size',
    type=int,
    default=256,
    help='batch size')

parser.add_argument(
    '--epochs',
    type=int,
    default=150,
    help='num of training epochs')

parser.add_argument(
    '--learning_rate',
    type=float,
    default=0.01,
    help='init learning rate')

parser.add_argument(
    '--lr_decay_step',
    default='50,100',
    type=str,
    help='learning rate')

parser.add_argument(
    '--momentum',
    type=float,
    default=0.9,
    help='momentum')

parser.add_argument(
    '--weight_decay',
    type=float,
    default=5e-4,
    help='weight decay')

parser.add_argument(
    '--resume',
    action='store_true',
    help='whether continue training from the same directory')

parser.add_argument(
    '--use_pretrain',
    action='store_true',
    help='whether use pretrain model')

parser.add_argument(
    '--pretrain_dir',
    type=str,
    default='',
    help='pretrain model path')

parser.add_argument(
    '--rank_conv_prefix',
    type=str,
    default='',
    help='rank conv file folder')

parser.add_argument(
    '--compress_rate',
    type=str,
    default=None,
    help='compress rate of each conv')

parser.add_argument(
    '--test_only',
    action='store_true',
    help='whether it is test mode')

parser.add_argument(
    '--test_model_dir',
    type=str,
    default='',
    help='test model path')

parser.add_argument(
    '--gpu',
    type=str,
    default='0',
    help='Select gpu to use')


parser.add_argument(
    '--num_class',
    type=int,
    default=10,
    help='number of classes')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
CLASSES = args.num_class
print_freq = (256*50)//args.batch_size

if not os.path.isdir(args.job_dir):
    os.makedirs(args.job_dir)

utils.record_config(args)
now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
logger = utils.get_logger(os.path.join(args.job_dir, 'logger'+now+'.log'))

def get_training_dataloader(mean, std, batch_size=64, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader

def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader


def data_loading_ISIC_train():
    transform_train = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.Resize(64),
        # transforms.RandomCrop(64, padding=0),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=45),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])
    my_data = torchvision.datasets.ImageFolder(root='./data/ISIC/ISIC-2017_Training_Data', transform=transform_train)
    tr_loader = DataLoader(my_data, shuffle=True, num_workers=2, batch_size=16)
    
    return tr_loader


def data_loading_ISIC_test():
    transform_test = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.Resize(64),
        # transforms.RandomCrop(64, padding=0),
        transforms.ToTensor()
    ])
    my_data = torchvision.datasets.ImageFolder(root='./data/ISIC/ISIC-2017_Validation_Data', transform=transform_test)
    te_loader = DataLoader(my_data, shuffle=True, num_workers=2, batch_size=16)
    
    return te_loader


def data_loading_IDRID_train():
    transform_train = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.Resize(64),
        # transforms.RandomCrop(64, padding=0),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(degrees=15),
        # transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])

    my_data = torchvision.datasets.ImageFolder(root='./data/IDRID/Macular Degeneration/1. Original Images/a. Training Set', transform=transform_train)
    tr_loader = DataLoader(my_data, shuffle=True, num_workers=2, batch_size=16)
    
    return tr_loader

def data_loading_IDRID_test():
    transform_test = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.Resize(64),
        # transforms.RandomCrop(64, padding=0),
        transforms.ToTensor()
    ])
    my_data = torchvision.datasets.ImageFolder(root='./data/IDRID/Macular Degeneration/1. Original Images/b. Testing Set', transform=transform_test)
    te_loader = DataLoader(my_data, shuffle=True, num_workers=2, batch_size=16)
    
    return te_loader

#use for loading pretrain model
if len(args.gpu)>1:
    name_base='module.'
else:
    name_base=''

def load_vgg_model(model, oristate_dict):
    state_dict = model.state_dict()
    last_select_index = None #Conv index selected in the previous layer

    cnt=0
    prefix = args.rank_conv_prefix+'/rank_conv'
    subfix = ".npy"
    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d):

            cnt+=1
            oriweight = oristate_dict[name + '.weight']
            curweight =state_dict[name_base+name + '.weight']
            orifilter_num = oriweight.size(0)
            currentfilter_num = curweight.size(0)

            if orifilter_num != currentfilter_num:

                cov_id = cnt
                logger.info('loading rank from: ' + prefix + str(cov_id) + subfix)
                rank = np.load(prefix + str(cov_id) + subfix)
                # print('rank',rank, rank.shape)
                select_index = np.argsort(rank)[orifilter_num-currentfilter_num:]  # preserved filter id
                select_index.sort()
                # print('select_index',select_index,select_index.shape)
                if last_select_index is not None:
                    for index_i, i in enumerate(select_index):
                        for index_j, j in enumerate(last_select_index):
                            # print('Overall',i,j,oristate_dict[name + '.weight'].shape,state_dict[name_base+name + '.weight'].shape)
                            state_dict[name_base+name + '.weight'][index_i][index_j] = \
                                oristate_dict[name + '.weight'][i][j]
                        # break
                else:
                    for index_i, i in enumerate(select_index):
                       state_dict[name_base+name + '.weight'][index_i] = \
                            oristate_dict[name + '.weight'][i]
                last_select_index = select_index

            elif last_select_index is not None:
                for i in range(orifilter_num):
                    for index_j, j in enumerate(last_select_index):
                        state_dict[name_base+name + '.weight'][i][index_j] = \
                            oristate_dict[name + '.weight'][i][j]
            else:
                state_dict[name_base+name + '.weight'] = oriweight
                last_select_index = None

    model.load_state_dict(state_dict)

def load_resnet_model(model, oristate_dict, layer):
    cfg = {
        56: [9, 9, 9],
        110: [18, 18, 18],
    }

    state_dict = model.state_dict()

    current_cfg = cfg[layer]
    last_select_index = None

    all_conv_weight = []

    prefix = args.rank_conv_prefix+'/rank_conv'
    subfix = ".npy"

    cnt=1
    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'
        for k in range(num):
            for l in range(2):

                cnt+=1
                cov_id=cnt

                conv_name = layer_name + str(k) + '.conv' + str(l + 1)
                conv_weight_name = conv_name + '.weight'
                all_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight =state_dict[name_base+conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                if orifilter_num != currentfilter_num:
                    logger.info('loading rank from: ' + prefix + str(cov_id) + subfix)
                    rank = np.load(prefix + str(cov_id) + subfix)
                    select_index = np.argsort(rank)[orifilter_num - currentfilter_num:]  # preserved filter id
                    select_index.sort()

                    if last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                    (1+rank[index_i])*oristate_dict[conv_weight_name][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[name_base+conv_weight_name][index_i] = \
                                (1+rank[index_i])*oristate_dict[conv_weight_name][i]

                    last_select_index = select_index

                elif last_select_index is not None:
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                (1+rank[index_i])*oristate_dict[conv_weight_name][index_i][j]
                    last_select_index = None

                else:
                    state_dict[name_base+conv_weight_name] = oriweight
                    last_select_index = None

    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if 'shortcut' in name:
                continue
            if conv_name not in all_conv_weight:
                state_dict[name_base+conv_name] = oristate_dict[conv_name]

        elif isinstance(module, nn.Linear):
            state_dict[name_base+name + '.weight'] = oristate_dict[name + '.weight']
            state_dict[name_base+name + '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)


def main():

    cudnn.benchmark = True
    cudnn.enabled=True
    logger.info("args = %s", args)

    if args.compress_rate:
        import re
        cprate_str = args.compress_rate
        cprate_str_list = cprate_str.split('+')
        pat_cprate = re.compile(r'\d+\.\d*')
        pat_num = re.compile(r'\*\d+')
        cprate = []
        for x in cprate_str_list:
            num = 1
            find_num = re.findall(pat_num, x)
            if find_num:
                print(find_num)
                assert len(find_num) == 1
                num = int(find_num[0].replace('*', ''))
            find_cprate = re.findall(pat_cprate, x)
            assert len(find_cprate) == 1
            cprate += [float(find_cprate[0])] * num

        compress_rate = cprate

    # load model
    logger.info('compress_rate:' + str(compress_rate))
    logger.info('==> Building model..')
    model = eval(args.arch)(compress_rate=compress_rate, num_classes=CLASSES).cuda()
    logger.info(model)

    #calculate model size



    # load training data
    if (args.dataset == 'cifar10'):
        input_image_size = 32
        train_loader, val_loader = cifar10.load_data(args)
        
    elif (args.dataset ==  'cifar100'):
        input_image_size = 32
        CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        train_loader = get_training_dataloader(
            CIFAR100_TRAIN_MEAN,
            CIFAR100_TRAIN_STD,
            num_workers=4,
            batch_size=args.batch_size,
            shuffle=True)

        val_loader = get_test_dataloader(
            CIFAR100_TRAIN_MEAN,
            CIFAR100_TRAIN_STD,
            num_workers=4,
            batch_size=128,
            shuffle=True
    )

    elif(args.dataset ==  'ISIC'):
        input_image_size = 256
        train_loader = data_loading_ISIC_train()
        val_loader = data_loading_ISIC_test()
    elif(args.dataset ==  'IDRID'): 
        input_image_size = 256
        train_loader = data_loading_IDRID_train()
        val_loader = data_loading_IDRID_test()
    else:
        print('please specify a dataset to train on!')
        raise NotImplementedError

    
    input_image = torch.randn(1, 3, input_image_size, input_image_size).cuda()
    flops, params = profile(model, inputs=(input_image,))
    logger.info('Params: %.2f' % (params))
    logger.info('Flops: %.2f' % (flops))
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    if args.test_only:
        if os.path.isfile(args.test_model_dir):
            logger.info('loading checkpoint {} ..........'.format(args.test_model_dir))
            checkpoint = torch.load(args.test_model_dir)
            model.load_state_dict(checkpoint['state_dict'])
            valid_obj, valid_top1_acc, valid_top5_acc = validate(0, val_loader, model, criterion, args)
        else:
            logger.info('please specify a checkpoint file')
        return

    if len(args.gpu) > 1:
        device_id = []
        for i in range((len(args.gpu) + 1) // 2):
            device_id.append(i)
        model = nn.DataParallel(model, device_ids=device_id).cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_decay_step = list(map(int, args.lr_decay_step.split(',')))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)

    start_epoch = 0
    best_top1_acc= 0

    # load the checkpoint if it exists
    checkpoint_dir = os.path.join(args.job_dir, 'checkpoint.pth.tar')
    if args.resume:
        logger.info('loading checkpoint {} ..........'.format(checkpoint_dir))
        checkpoint = torch.load(checkpoint_dir)
        start_epoch = checkpoint['epoch'] + 1
        best_top1_acc = checkpoint['best_top1_acc']

        # deal with the single-multi GPU problem
        new_state_dict = OrderedDict()
        tmp_ckpt = checkpoint['state_dict']
        if len(args.gpu) > 1:
            for k, v in tmp_ckpt.items():
                new_state_dict['module.' + k.replace('module.', '')] = v
        else:
            for k, v in tmp_ckpt.items():
                new_state_dict[k.replace('module.', '')] = v

        model.load_state_dict(new_state_dict)
        logger.info("loaded checkpoint {} epoch = {}".format(checkpoint_dir, checkpoint['epoch']))
    else:
        if args.use_pretrain:
            logger.info('resuming from pretrain model')
            origin_model = eval(args.arch)(compress_rate=[0.] * 100, num_classes=CLASSES).cuda()
            ckpt = torch.load(args.pretrain_dir, map_location='cuda:0')

            #if args.arch=='resnet_56':
            #    origin_model.load_state_dict(ckpt['state_dict'],strict=False)
            if args.arch == 'resnet_110':
                new_state_dict = OrderedDict()
                for k, v in ckpt['state_dict'].items():
                    new_state_dict[k.replace('module.', '')] = v
                origin_model.load_state_dict(new_state_dict)
            else:
                origin_model.load_state_dict(ckpt['state_dict'])

            oristate_dict = origin_model.state_dict()

            if args.arch == 'vgg_16_bn':
                load_vgg_model(model, oristate_dict)
            elif args.arch == 'resnet_56':
                load_resnet_model(model, oristate_dict, 56)
            elif args.arch == 'resnet_110':
                load_resnet_model(model, oristate_dict, 110)
            else:
                raise
        else:
            print('training from scratch')

    # adjust the learning rate according to the checkpoint
    for epoch in range(start_epoch):
        scheduler.step()

    # train the model
    epoch = start_epoch
    while epoch < args.epochs:
        train_obj, train_top1_acc,  train_top5_acc = train(epoch,  train_loader, model, criterion, optimizer, scheduler)
        valid_obj, valid_top1_acc, valid_top5_acc = validate(epoch, val_loader, model, criterion, args)

        is_best = False
        if valid_top1_acc > best_top1_acc:
            best_top1_acc = valid_top1_acc
            is_best = True

        utils.save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_top1_acc': best_top1_acc,
            'optimizer' : optimizer.state_dict(),
            }, is_best, args.job_dir)

        epoch += 1
        logger.info("=>Best accuracy {:.3f}".format(best_top1_acc))#


def train(epoch, train_loader, model, criterion, optimizer, scheduler):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')

    model.train()
    end = time.time()

    for param_group in optimizer.param_groups:
        cur_lr = param_group['lr']
    logger.info('learning_rate: ' + str(cur_lr))

    num_iter = len(train_loader)
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.cuda()
        target = target.cuda()
        # compute outputy
        logits = model(images)
        loss = criterion(logits, target)

        # measure accuracy and record loss
        prec1, prec5 = utils.accuracy(logits, target, topk=(1,2))
        n = images.size(0)
        losses.update(loss.item(), n)   #accumulated loss
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            logger.info(
                'Epoch[{0}]({1}/{2}): '
                'Loss {loss.avg:.4f} '
                'Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}'.format(
                    epoch, i, num_iter, loss=losses,
                    top1=top1, top5=top5))

    scheduler.step()

    return losses.avg, top1.avg, top5.avg

def validate(epoch, val_loader, model, criterion, args):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')
    auc1 = utils.AverageMeter('auc', ':6.2f')
    my_f1_s1 = utils.AverageMeter('f1-score', ':6.2f')
    my_prec1 = utils.AverageMeter('precision', ':6.2f')
    my_spec1 = utils.AverageMeter('specificity', ':6.2f')
    my_reca1 = utils.AverageMeter('recall', ':6.2f')

    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            # compute output
            logits = model(images)
            loss = criterion(logits, target)

            # measure accuracy and record loss
            pred1, pred5 = utils.accuracy(logits, target, topk=(1, 2))
            auc, my_f1_s, my_prec, my_spec, my_reca = utils.other_metrics(logits, target, topk=(1), num_classes=CLASSES)

            n = images.size(0)
            losses.update(loss.item(), n)
            top1.update(pred1[0], n)
            top5.update(pred5[0], n)
            auc1.update(auc, n)
            my_f1_s1.update(my_f1_s, n)
            my_prec1.update(my_prec, n)
            my_spec1.update(my_spec, n)
            my_reca1.update(my_reca, n)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        logger.info(' * Acc@1:{top1.avg:.3f} - Acc@5:{top5.avg:.3f} - AUC:{auc1.avg:.3f} - F1-score:{my_f1_s1.avg:.3f} - Precision:{my_prec1.avg:.3f} - Specificity:{my_spec1.avg:.3f} - Recall:{my_reca1.avg:.3f}'
              .format(top1=top1, top5=top5, auc1=auc1, my_f1_s1=my_f1_s1, my_prec1=my_prec1, my_spec1=my_spec1, my_reca1=my_reca1))

    return losses.avg, top1.avg, top5.avg


if __name__ == '__main__':
  main()
