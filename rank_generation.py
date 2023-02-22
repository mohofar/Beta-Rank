
import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision
from torchvision import datasets, transforms

from models.vgg import vgg_16_bn
from models.resnet import resnet_56,resnet_110
from tqdm import tqdm
# from data import imagenet, imagenet_dali
from torch.utils.data import DataLoader

import utils.common as utils

parser = argparse.ArgumentParser(description='Rank extraction')

parser.add_argument(
    '--data_dir',
    type=str,
    default='./data',
    help='dataset path')
parser.add_argument(
    '--dataset',
    type=str,
    default='cifar10',
    choices=('cifar10', 'cifar100', 'ISIC', 'IDRID'),
    help='dataset')

parser.add_argument(
    '--pruning_method',
    type=str,
    default='Beta',
    choices=('Beta', 'Hrank', 'L1'),
    help='methods of pruning (Beta-rank, Hrank, L1-Norm)')

parser.add_argument(
    '--job_dir',
    type=str,
    default='result/tmp',
    help='The directory where the summaries will be stored.')
parser.add_argument(
    '--arch',
    type=str,
    default='vgg_16_bn',
    choices=('vgg_16_bn','resnet_56','resnet_110'),
    help='The architecture to prune')
parser.add_argument(
    '--pretrain_dir',
    type=str,
    default=None,
    help='load the model from the specified checkpoint')
parser.add_argument(
    '--limit',
    type=int,
    default=1,
    help='The num of batch to get rank.')

parser.add_argument(
    '--num_class',
    type=int,
    default=10,
    help='number of classes')
parser.add_argument(
    '--batch_size',
    type=int,
    default=256,
    help='Batch size for training.')
parser.add_argument(
    '--gpu',
    type=str,
    default='0',
    help='Select gpu to use')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    my_data = torchvision.datasets.ImageFolder(root='./data/ISIC/ISIC-2017_Training_Data', transform=transform_train)
    tr_loader = DataLoader(my_data, shuffle=True, num_workers=2, batch_size=16)
    
    return tr_loader

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

    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    my_data = torchvision.datasets.ImageFolder(root='./data/IDRID/Macular Degeneration/1. Original Images/a. Training Set', transform=transform_train)
    tr_loader = DataLoader(my_data, shuffle=True, num_workers=2, batch_size=16)
    
    return tr_loader

# Data
print('==> Preparing data..')
if args.dataset=='cifar10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

elif args.dataset=='cifar100':
    CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    train_loader = get_training_dataloader(
        CIFAR100_TRAIN_MEAN,
        CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.batch_size,
        shuffle=True
    )

elif args.dataset=='ISIC':
    train_loader = data_loading_ISIC_train()

elif args.dataset=='IDRID':
    train_loader = data_loading_IDRID_train()

else:
    print('Please specify a detaset!')
    raise NotImplementedError

# Model
print('==> Building model..')
net = eval(args.arch)(compress_rate=[0.]*100, num_classes=args.num_class)
net = net.to(device)
print(net)

if len(args.gpu)>1 and torch.cuda.is_available():
    device_id = []
    for i in range((len(args.gpu) + 1) // 2):
        device_id.append(i)
    net = torch.nn.DataParallel(net, device_ids=device_id)

if args.pretrain_dir:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(args.pretrain_dir, map_location='cuda:'+args.gpu)
    net.load_state_dict(checkpoint['state_dict'])
else:
    print('please speicify a pretrain model ')
    raise NotImplementedError

criterion = nn.CrossEntropyLoss()
feature_result = torch.tensor(0.)
total = torch.tensor(0.)

#get feature map of certain layer via hook
def get_feature_hook(self, input, output):
    global feature_result
    global entropy
    global total
    a = output.shape[0]
    b = output.shape[1]
    c = torch.tensor([torch.linalg.matrix_rank(output[i,j,:,:]).item() for i in range(a) for j in range(b)])

    c = c.view(a, -1).float()
    c = c.sum(0)
    feature_result = feature_result * total + c
    total = total + a
    feature_result = feature_result / total

def std_filters(inp, out, filters):
    num_filters = out.shape[2]*out.shape[3]
    stride = int(inp.shape[3]/out.shape[3])
    filters_rank = []
    
    for out_ch in tqdm(range(out.shape[1])): #64
        filter = np.zeros(filters.shape)
        for out_dim1 in (range(out.shape[2])): #16
            for out_dim2 in (range(out.shape[3])): #16
                
                a = np.std(out[:,out_ch,out_dim1,out_dim2],axis=0) #out std
                b = (np.std(inp[:,out_dim1*stride:out_dim1*stride+filters.shape[2],
                                         out_dim2*stride:out_dim2*stride+filters.shape[3],:])) #inp std
                if(str(b)!='nan'):
                    filter += np.abs(filters[out_ch]*a/b)
        filters_rank.append(np.mean(filter))
    filters_rank = (filters_rank - np.amin(filters_rank))/(np.amax(filters_rank)-np.amin(filters_rank))
    return np.array(filters_rank)

def get_feature_hook_2(self, input, output):
    global feature_result
    global entropy
    global total
    global my_input
    global my_output

    my_output = np.copy(output.cpu().detach().numpy())
    my_input = np.copy(input[0].cpu().detach().numpy())
    
def l1_filters(inp, out, filters):
    # print('-->',inp.shape,out.shape,filters.shape)
    num_filters = out.shape[2]*out.shape[3]
    stride = int(inp.shape[3]/out.shape[3])
    filters_rank = []
    

    for out_ch in tqdm(range(out.shape[1])): #64
        filter = np.zeros(filters.shape)
        for out_dim1 in (range(out.shape[2])): #16
            for out_dim2 in (range(out.shape[3])): #16
                
                a = 1
                b = 1
                if(str(b)=='nan'):
                    break
                filter += np.abs(filters[out_ch]*a/b)
        filters_rank.append(np.mean(filter))
    return np.array(filters_rank)


def inference():
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    limit = args.limit

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if batch_idx >= limit:
               break

            inputs, targets = inputs.to(device), targets.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            utils.progress_bar(batch_idx, limit, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))#'''

if args.arch=='vgg_16_bn'and args.pruning_method=='Hrank':

    if len(args.gpu) > 1:
        relucfg = net.module.relucfg
    else:
        relucfg = net.relucfg
    # relucfg = [0] + relucfg
    for i, cov_id in enumerate(relucfg):
        cov_layer = net.features[cov_id]
        handler = cov_layer.register_forward_hook(get_feature_hook)
        inference()
        handler.remove()

        if not os.path.isdir('rank_conv/'+args.arch+'_'+args.pruning_method+'_'+args.dataset+'_limit%d'%(args.limit)):
            os.mkdir('rank_conv/'+args.arch+'_'+args.pruning_method+'_'+args.dataset+'_limit%d'%(args.limit))
        np.save('rank_conv/'+args.arch+'_'+args.pruning_method+'_'+args.dataset+'_limit%d'%(args.limit)+'/rank_conv' + str(i + 1) + '.npy', feature_result.numpy())

        feature_result = torch.tensor(0.)
        total = torch.tensor(0.)

elif args.arch=='vgg_16_bn' and args.pruning_method=='Beta':

    if len(args.gpu) > 1:
        relucfg = net.module.relucfg
    else:
        relucfg = net.relucfg
    relucfg =  [-1]+relucfg
    for i, cov_id in enumerate(relucfg[:-1]):
        cov_layer = net.features[cov_id+1]
        handler = cov_layer.register_forward_hook(get_feature_hook_2)
        inference()
        handler.remove()
        filters = net.features[cov_id+1].weight.cpu().detach().numpy()
        feature_result = std_filters(my_input, my_output,filters)
        if not os.path.isdir('rank_conv/'+args.arch+'_'+args.pruning_method+'_'+args.dataset+'_limit%d'%(args.limit)):
            os.mkdir('rank_conv/'+args.arch+'_'+args.pruning_method+'_'+args.dataset+'_limit%d'%(args.limit))
        np.save('rank_conv/'+args.arch+'_'+args.pruning_method+'_'+args.dataset+'_limit%d'%(args.limit)+'/rank_conv' + str(i + 1) + '.npy', feature_result)
        feature_result = torch.tensor(0.)
        total = torch.tensor(0.)

elif args.arch=='vgg_16_bn' and args.pruning_method=='L1':

    if len(args.gpu) > 1:
        relucfg = net.module.relucfg
    else:
        relucfg = net.relucfg
    relucfg =  [-1]+relucfg
    for i, cov_id in enumerate(relucfg[:-1]):
        cov_layer = net.features[cov_id+1]
        handler = cov_layer.register_forward_hook(get_feature_hook_2)
        inference()
        handler.remove()
        filters = net.features[cov_id+1].weight.cpu().detach().numpy()
        feature_result = l1_filters(my_input, my_output,filters)
        if not os.path.isdir('rank_conv/'+args.arch+'_'+args.pruning_method+'_'+args.dataset+'_limit%d'%(args.limit)):
            os.mkdir('rank_conv/'+args.arch+'_'+args.pruning_method+'_'+args.dataset+'_limit%d'%(args.limit))
        np.save('rank_conv/'+args.arch+'_'+args.pruning_method+'_'+args.dataset+'_limit%d'%(args.limit)+'/rank_conv' + str(i + 1) + '.npy', feature_result)
        feature_result = torch.tensor(0.)
        total = torch.tensor(0.)

elif args.arch=='resnet_56' and args.pruning_method=='Hrank':

    cov_layer = eval('net.relu')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    if not os.path.isdir('rank_conv/' + args.arch+'_'+args.pruning_method+'_'+args.dataset+'_limit%d'%(args.limit)):
        os.mkdir('rank_conv/' + args.arch+'_'+args.pruning_method+'_'+args.dataset+'_limit%d'%(args.limit))
    np.save('rank_conv/' + args.arch+'_'+args.pruning_method+'_'+args.dataset+'_limit%d'%(args.limit)+ '/rank_conv%d' % (1) + '.npy', feature_result.numpy())
    feature_result = torch.tensor(0.)
    total = torch.tensor(0.)

    # ResNet56 per block
    cnt=1
    for i in range(3):
        block = eval('net.layer%d' % (i + 1))
        for j in range(9):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            np.save('rank_conv/' + args.arch +'_'+args.pruning_method+'_'+args.dataset+'_limit%d'%(args.limit)+ '/rank_conv%d'%(cnt + 1)+'.npy', feature_result.numpy())
            cnt+=1
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            np.save('rank_conv/' + args.arch+'_'+args.pruning_method+'_'+args.dataset+'_limit%d'%(args.limit)+ '/rank_conv%d'%(cnt + 1)+'.npy', feature_result.numpy())
            cnt += 1
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)

elif args.arch=='resnet_56' and args.pruning_method=='Beta':
    print('Rank generation...')
    with torch.no_grad():

        cov_layer = eval('net.conv1')
        handler = cov_layer.register_forward_hook(get_feature_hook_2)
        inference()
        handler.remove()
        filters = cov_layer.weight.cpu().detach().numpy()
        feature_result = std_filters(my_input, my_output,filters)
        if not os.path.isdir('rank_conv/'+args.arch+'_'+args.pruning_method+'_'+args.dataset+'_limit%d'%(args.limit)):
            os.mkdir('rank_conv/'+args.arch+'_'+args.pruning_method+'_'+args.dataset+'_limit%d'%(args.limit))
        np.save('rank_conv/'+args.arch+'_'+args.pruning_method+'_'+args.dataset+'_limit%d'%(args.limit)+'/rank_conv' + str(1) + '.npy', feature_result)
        feature_result = torch.tensor(0.)
        total = torch.tensor(0.)
        # ResNet56 per block
        cnt=1
        for i in range(3):
            block = eval('net.layer%d' % (i + 1))
            for j in range(9):
                cov_layer = block[j].conv1
                handler = cov_layer.register_forward_hook(get_feature_hook_2)
                inference()
                handler.remove()
                filters = cov_layer.weight.cpu().detach().numpy()
                feature_result = std_filters(my_input, my_output,filters)
                np.save('rank_conv/'+args.arch+'_'+args.pruning_method+'_'+args.dataset+'_limit%d'%(args.limit)+'/rank_conv' + str(cnt + 1) + '.npy', feature_result)

                cnt+=1
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

                cov_layer = block[j].conv2
                handler = cov_layer.register_forward_hook(get_feature_hook_2)
                inference()
                handler.remove()
                filters = cov_layer.weight.cpu().detach().numpy()
                feature_result = std_filters(my_input, my_output,filters)
                np.save('rank_conv/'+args.arch+'_'+args.pruning_method+'_'+args.dataset+'_limit%d'%(args.limit)+'/rank_conv' + str(cnt + 1) + '.npy', feature_result)

                cnt += 1
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

elif args.arch=='resnet_56' and args.pruning_method=='L1':
    print('Rank generation...')
    with torch.no_grad():

        cov_layer = eval('net.conv1')
        handler = cov_layer.register_forward_hook(get_feature_hook_2)
        inference()
        handler.remove()
        filters = cov_layer.weight.cpu().detach().numpy()
        feature_result = l1_filters(my_input, my_output,filters)
        if not os.path.isdir('rank_conv/'+args.arch+'_'+args.pruning_method+'_'+args.dataset+'_limit%d'%(args.limit)):
            os.mkdir('rank_conv/'+args.arch+'_'+args.pruning_method+'_'+args.dataset+'_limit%d'%(args.limit))
        np.save('rank_conv/'+args.arch+'_'+args.pruning_method+'_'+args.dataset+'_limit%d'%(args.limit)+'/rank_conv' + str(1) + '.npy', feature_result)
        feature_result = torch.tensor(0.)
        total = torch.tensor(0.)
        # ResNet56 per block
        cnt=1
        for i in range(3):
            block = eval('net.layer%d' % (i + 1))
            for j in range(9):
                cov_layer = block[j].conv1
                handler = cov_layer.register_forward_hook(get_feature_hook_2)
                inference()
                handler.remove()
                filters = cov_layer.weight.cpu().detach().numpy()
                feature_result = l1_filters(my_input, my_output,filters)
                np.save('rank_conv/'+args.arch+'_'+args.pruning_method+'_'+args.dataset+'_limit%d'%(args.limit)+'/rank_conv' + str(cnt + 1) + '.npy', feature_result)

                cnt+=1
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

                cov_layer = block[j].conv2
                handler = cov_layer.register_forward_hook(get_feature_hook_2)
                inference()
                handler.remove()
                filters = cov_layer.weight.cpu().detach().numpy()
                feature_result = l1_filters(my_input, my_output,filters)
                np.save('rank_conv/'+args.arch+'_'+args.pruning_method+'_'+args.dataset+'_limit%d'%(args.limit)+'/rank_conv' + str(cnt + 1) + '.npy', feature_result)

                cnt += 1
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

elif args.arch=='resnet_110' and args.pruning_method=='Hrank':

    cov_layer = eval('net.relu')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    if not os.path.isdir('rank_conv/' + args.arch+'_'+args.pruning_method+'_'+args.dataset+'_limit%d'%(args.limit)):
        os.mkdir('rank_conv/' + args.arch+'_'+args.pruning_method+'_'+args.dataset+'_limit%d'%(args.limit))
    np.save('rank_conv/' + args.arch+'_'+args.pruning_method+'_'+args.dataset+'_limit%d'%(args.limit) + '/rank_conv%d' % (1) + '.npy', feature_result.numpy())
    feature_result = torch.tensor(0.)
    total = torch.tensor(0.)

    cnt = 1
    # ResNet110 per block
    for i in range(3):
        block = eval('net.layer%d' % (i + 1))
        for j in range(18):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            np.save('rank_conv/' + args.arch  + '_'+args.pruning_method+'_'+args.dataset+'_limit%d' % (args.limit) + '/rank_conv%d' % (
            cnt + 1) + '.npy', feature_result.numpy())
            cnt += 1
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            np.save('rank_conv/' + args.arch  + '_'+args.pruning_method+'_'+args.dataset+'_limit%d' % (args.limit) + '/rank_conv%d' % (
                cnt + 1) + '.npy', feature_result.numpy())
            cnt += 1
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)

elif args.arch=='resnet_110'and args.pruning_method=='Beta':

    cov_layer = eval('net.conv1')
    handler = cov_layer.register_forward_hook(get_feature_hook_2)
    inference()
    handler.remove()
    filters = cov_layer.weight.cpu().detach().numpy()
    feature_result = std_filters(my_input, my_output,filters)
    if not os.path.isdir('rank_conv/'+args.arch+'_'+args.pruning_method+'_'+args.dataset+'_limit%d'%(args.limit)):
        os.mkdir('rank_conv/'+args.arch+'_'+args.pruning_method+'_'+args.dataset+'_limit%d'%(args.limit))
    np.save('rank_conv/'+args.arch+'_'+args.pruning_method+'_'+args.dataset+'_limit%d'%(args.limit)+'/rank_conv' + str(1) + '.npy', feature_result)
    feature_result = torch.tensor(0.)
    total = torch.tensor(0.)

    cnt = 1
    # ResNet110 per block
    for i in range(3):
        block = eval('net.layer%d' % (i + 1))
        for j in range(18):
            cov_layer = block[j].conv1
            handler = cov_layer.register_forward_hook(get_feature_hook_2)
            inference()
            handler.remove()
            filters = cov_layer.weight.cpu().detach().numpy()
            feature_result = std_filters(my_input, my_output,filters)
            np.save('rank_conv/'+args.arch+'_'+args.pruning_method+'_'+args.dataset+'_limit%d'%(args.limit)+'/rank_conv' + str(cnt + 1) + '.npy', feature_result)

            cnt+=1
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)

            cov_layer = block[j].conv2
            handler = cov_layer.register_forward_hook(get_feature_hook_2)
            inference()
            handler.remove()
            filters = cov_layer.weight.cpu().detach().numpy()
            feature_result = std_filters(my_input, my_output,filters)
            np.save('rank_conv/'+args.arch+'_'+args.pruning_method+'_'+args.dataset+'_limit%d'%(args.limit)+'/rank_conv' + str(cnt + 1) + '.npy', feature_result)

            cnt += 1
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)

elif args.arch=='resnet_110'and args.pruning_method=='L1':

    cov_layer = eval('net.conv1')
    handler = cov_layer.register_forward_hook(get_feature_hook_2)
    inference()
    handler.remove()
    filters = cov_layer.weight.cpu().detach().numpy()
    feature_result = l1_filters(my_input, my_output,filters)
    if not os.path.isdir('rank_conv/'+args.arch+'_'+args.pruning_method+'_'+args.dataset+'_limit%d'%(args.limit)):
        os.mkdir('rank_conv/'+args.arch+'_'+args.pruning_method+'_'+args.dataset+'_limit%d'%(args.limit))
    np.save('rank_conv/'+args.arch+'_'+args.pruning_method+'_'+args.dataset+'_limit%d'%(args.limit)+'/rank_conv' + str(1) + '.npy', feature_result)
    feature_result = torch.tensor(0.)
    total = torch.tensor(0.)

    cnt = 1
    # ResNet110 per block
    for i in range(3):
        block = eval('net.layer%d' % (i + 1))
        for j in range(18):
            cov_layer = block[j].conv1
            handler = cov_layer.register_forward_hook(get_feature_hook_2)
            inference()
            handler.remove()
            filters = cov_layer.weight.cpu().detach().numpy()
            feature_result = l1_filters(my_input, my_output,filters)
            np.save('rank_conv/'+args.arch+'_'+args.pruning_method+'_'+args.dataset+'_limit%d'%(args.limit)+'/rank_conv' + str(cnt + 1) + '.npy', feature_result)

            cnt+=1
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)

            cov_layer = block[j].conv2
            handler = cov_layer.register_forward_hook(get_feature_hook_2)
            inference()
            handler.remove()
            filters = cov_layer.weight.cpu().detach().numpy()
            feature_result = l1_filters(my_input, my_output,filters)
            np.save('rank_conv/'+args.arch+'_'+args.pruning_method+'_'+args.dataset+'_limit%d'%(args.limit)+'/rank_conv' + str(cnt + 1) + '.npy', feature_result)

            cnt += 1
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)

#'''