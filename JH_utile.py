# -*- coding: utf-8 -*-
# from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import _pickle as cPickle
import JH_error_list
from PIL import Image

from networks import preact_resnet          as PRN
from networks import preact_resnet_eeg          as PRNEEG
from networks import preact_resnet_covid    as PRNC

# from networks import preact_resnet_test as PRNT


import copy
import sys
import os

import math



## ======================================================##
## ============ function for Custom Dataset =============##
## ======================================================##
class CovidDataSet(Dataset):
    def __init__(self, data, labels, data_max, data_min, transform=None, target_transform=None):
        self.labels = labels
        self.data = data
        self.transform = transform
        self.target_transform = target_transform
        self.data_max = data_max
        self.data_min = data_min

    def __getitem__(self, index):
        covid_data = self.data[index]
        target = self.labels[index]

        # to return a Covid data (normalized 0 to 1)
        pos_covid_data = covid_data - self.data_min
        nor_covid_data = pos_covid_data/(self.data_max-self.data_min)

        # shape change H x w x C --> C x H x W
        # permuted_nor_eeg_data = nor_eeg_data.permute(2,0,1)

        if self.transform is not None:
            nor_eeg_data = self.transform(nor_covid_data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return nor_covid_data, target

    def __len__(self):
        return len(self.data)


def data_load(data_ori,args,nor=False):

    num_classes = len(args.class_list)
    dataset = copy.deepcopy(data_ori)
    # dataset_info = torch.load(dataset_path + 'data_list_dict.pt')

    if nor == True:
        train_data  = dataset['train_data_nor']
        test_data   = dataset['test_data_nor']
    elif nor == False:
        train_data = dataset['train_data']
        test_data  = dataset['test_data']
    else:
        raise JH_error_list.Error_nor

    train_labels    = dataset['train_label']
    test_labels     = dataset['test_label']



    return train_data, train_labels, test_data, test_labels, num_classes


def load_nor_value(data_ori):
    dataset = copy.deepcopy(data_ori)
    data_max = dataset['max_value']
    data_min = dataset['min_value']
    train_data_mean = dataset['mean_train_nor']
    train_data_std = dataset['std_train_nor']
    test_data_mean = dataset['mean_test_nor']
    test_data_std = dataset['std_test_nor']


    return data_max, data_min, train_data_mean, train_data_std, test_data_mean, test_data_std






## ======================================================##
## ===================== functions ======================##
## ======================================================##

def print_Net_Setting(args):
    print('Data_set\t: {DB}'.format(DB=args.dataset))
    print('Epoch\t\t: {E}'.format(E=args.num_epochs))
    print('Lr_Drop \t: {LD}'.format(LD=args.lr_drop_epoch))
    print('Net_type\t: {NT}'.format(NT=args.net_type))
    print('Depth\t\t: {D}'.format(D=args.depth))

    if hasattr(args, 'widen_factor'):
        print('WidenFactor\t: {WF}'.format(WF=args.widen_factor))
    if hasattr(args, 'alpha'):
        print('Alpha\t\t: {a}'.format(a=args.alpha))
    if hasattr(args, 'pl'):
        print('Pl\t\t\t: {P}'.format(P=args.pl))
    if hasattr(args, 'bottleneck'):
        print('Bottle Neck\t: {BN}'.format(BN=args.bottleneck))
    if hasattr(args, 'lr'):
        print('Initial lr\t: {lr}'.format(lr=args.lr))
    if hasattr(args, 'batch_size'):
        print('Batch size\t: {BS}'.format(BS=args.batch_size))
    if hasattr(args, 'cutout'):
        print('Cutout\t\t: {C}'.format(C=args.cutout))
    if hasattr(args, 'nesterov'):
        print('nesterov\t: {N}'.format(N=args.nesterov))



def make_Dir(base_path, path_list):

    # file_name = './'
    file_name = base_path
    for i, list in enumerate(path_list):
        file_name = file_name + list
        if not os.path.isdir(file_name):
            os.mkdir(file_name)
        file_name = file_name + os.sep
    return file_name


def topk_accurcy(outputs, targets, topk_range=(1,)):
    maxk = max(topk_range)
    batch_size = targets.size(0)

    _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)
    # _,pred = torch.topk(outputs, maxk, dim=1, largest=True, sorted=True) # 위와 같은 코드임
    pred_t = pred.t()

    target_flat = targets.view(1, -1).expand(pred_t.size())
    # self.expand_as(other) is equivalent to self.expand(other.size())
    correct = pred_t.eq(target_flat)

    topk_acc = []
    for k in topk_range:
        correct_k = correct[:k].reshape(-1).float().sum()
        correct_k = correct_k.data.item()
        topk_acc.append(correct_k / batch_size * 100.)

    return topk_acc




class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def meanstdEEG(data_list):

    data = []
    mean = []
    std  = []

    for for_i in data_list:
        data.append(torch.load(for_i).numpy())
        print(for_i)
    # data=torch.stack(tuple(data))
    data = np.array(data)

    ch = data.shape[3]
    for for_i in range(ch):
        print(for_i)
        # data_ch = data[:,:,:,for_i]
        mean.append(data[:,:,:,for_i].mean())
        std.append(data[:,:,:,for_i].std())
        # torch gkatn 쓸경우 unbiased=False
    return np.array(mean), np.array(std)


class ConfusionMatrix(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.reset()

    def reset(self):
        self.conf_matrix = np.zeros((self.num_class, self.num_class))
        self.conf_matrix_avg = np.zeros((self.num_class, self.num_class))
        # self.count       = 0.
        self.count = np.zeros(self.num_class)


    # row: label(Ground Truth = input label), column: result(pred),
    def update(self, label, pred, n=1):
        result = pred.max(dim=1)
        result = result.indices

        for for_i in label:
            self.count[for_i] += 1


        # self.count += n
        for for_i in range(n):
            self.conf_matrix[label[for_i]][result[for_i]] += 1

        for for_i in range(self.num_class):
            for for_j in range(self.num_class):
                self.conf_matrix_avg[for_i][for_j] = self.conf_matrix[for_i][for_j]/self.count[for_i]





def meanstd(data):
    ch = data.shape[3]

    mean = []
    std = []
    for i in range(ch):
        data_ch = data[:,:,:,i]
        mean.append(np.mean(data_ch))
        std.append(np.std(data_ch))
    return np.array(mean), np.array(std)

def meanstd2(data):
    return np.mean(data), np.std(data)


def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s


def imshow(img):
    npimg = img     # unnormalize
    # npimg = img.numpy()
    # plt.imshow(np.transpose(npimg, (1, 2, 0))) #img : channel, height, width --> height, width, channel 순서 변경
    plt.imshow(npimg)
    plt.show()




def param(num_classes, num_groups):

    param = torch.Tensor(num_classes, num_groups).normal_(mean=0, std=0.01)
    # GPU 에 올릴지 말지 수정 필요
    return nn.Parameter(param, requires_grad=True)


def class_score_to_list(net_outputs, target_label):
    data_class_score = []
    data_label = []

    class_score = F.softmax(net_outputs, dim=1)
    data_class_score2 = class_score.cpu()
    data_label2 = target_label.cpu()
    for i, score in enumerate(class_score):
        data_class_score.append(score.cpu())

    for i, label in enumerate(target_label):
        data_label.append(label.cpu())

    # data_label_list = []
    # for i in range(len(data_label)):
    #     temp_label = data_label[i]
    #     data_label_list.append(temp_label.data.item())
    # data_label = data_label_list

    data_class_score = torch.stack(tuple(data_class_score))

    return data_class_score, data_label




# def error_message_network_name():
#     print('Error : Network should be either [preact_resnet / preact_resnet_EEG]')


def get_filename(args, num_classes, group_label_info=None):
    num_group = num_classes
    str_depth = '-D' + '_{D:02d}'.format(D=args.depth)
    file_name = args.net_type + str_depth

    if hasattr(args, 'bottleneck'):
        file_name = file_name + '_bottleneck'

    return file_name


def get_network_init_weight(args, num_classes):
    if args.net_type == 'preact_resnet':
        net = PRN.PreActResNet(args.depth, num_classes, args.dataset, bottleneck=args.bottleneck)
        net.apply(PRN.weight_init)

    elif args.net_type == 'preact_resnet_covid':
        net = PRNC.PreActResNet(args.depth, num_classes, args.dataset, bottleneck=args.bottleneck)
        net.apply(PRNC.weight_init)


    else:
        raise JH_error_list.Error_net_name

    return net


def learning_rate(args, epoch):

    init_lr = args.lr
    lr = init_lr
    gamma = args.gamma
    lr_drop_epoch = args.lr_drop_epoch
    for i in range(len(lr_drop_epoch)):
        if epoch >= lr_drop_epoch[i]:
            lr = init_lr * math.pow(gamma, i + 1)

    return lr

def momentum_weightdecay(args):
    if args.net_type == 'preact_resnet':
        mmt = float(0.9)
        w_decay = float(1e-4)

    elif args.net_type == 'preact_resnet_covid':
        mmt = float(0.9)
        w_decay = float(1e-4)

    else:
        # error_message_network_name()
        raise JH_error_list.Error_mometum_weight


    return mmt, w_decay




def info_plot_acc(input_args):
    info = {
            'net_type' : input_args.net_type
    }


def save_network(input_args, state_dict, best_top1_acc, epoch, summary, group_label_info=None):
    args = copy.deepcopy(input_args)
    state = {
            'args'          : args,
            'state_dict'    : state_dict,
            'best_top1_acc' : best_top1_acc,
            'epoch'         : epoch,
            'summary'       : summary,
            'group_label_info': group_label_info
            }


    return state
# def load_param(input_args, checkpoint):
#     # Load checkpoint data
#     args = copy.deepcopy(input_args)
#     print("| Resuming from checkpoint...")
#
#     args                = checkpoint['args']
#     args.start_epoch    = checkpoint['epoch'] + 1
#     state_dict          = checkpoint['state_dict']
#     best_top1_acc       = checkpoint['best_top1_acc']
#     best_top5_acc       = checkpoint['best_top5_acc']
#     summary             = checkpoint['summary']
#     plot_acc_info       = checkpoint['plot_acc_info']
#
#
#     return args, state_dict, best_top1_acc, best_top5_acc, summary, plot_acc_info

def load_param(input_args, checkpoint):
    # Load checkpoint data
    # temp_args = copy.deepcopy(input_args)
    print("| Resuming from checkpoint...")
    # if

    args                = checkpoint['args']
    if hasattr(input_args, 'testOnly'):
        args.testOnly   = input_args.testOnly
    args.start_epoch    = checkpoint['epoch'] + 1
    state_dict          = checkpoint['state_dict']
    best_top1_acc       = checkpoint['best_top1_acc']
    summary             = checkpoint['summary']

    return args, state_dict, best_top1_acc, summary


def save_class_score(net, dataloader, num_dataset, device):

    data_class_score = []
    data_label = []

    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            class_score = F.softmax(outputs, dim=1)

            for i, score in enumerate(class_score):
                data_class_score.append(score.cpu())

            for i, label in enumerate(targets):
                data_label.append(label.cpu())

            print('| Progress: Iter[{current_idx_set:3d} / {total_num_set:3d}]'
                  .format(current_idx_set=batch_idx + 1, total_num_set=math.ceil((num_dataset/dataloader.batch_size))))

        data_class_score = torch.stack(tuple(data_class_score))
        # print(data_class_score)

    return data_class_score, data_label


def save_class_score_without_softmax(net, dataloader, num_dataset, device):

    data_class_score = []
    data_label = []

    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            for i, score in enumerate(outputs):
                data_class_score.append(score.cpu())

            for i, label in enumerate(targets):
                data_label.append(label.cpu())

            print('| Progress: Iter[{current_idx_set:3d} / {total_num_set:3d}]'
                  .format(current_idx_set=batch_idx + 1, total_num_set=math.ceil((num_dataset/dataloader.batch_size))))

        data_class_score = torch.stack(tuple(data_class_score))

    return data_class_score, data_label

def network_hyperparameter_check(args, args2, lr_drop_revise=False):
    print('Checking network hyparameter......')
    if args.depth == args2.depth:
        assert args.depth           == args2.depth,         'Please check network depth'

    assert args.num_epochs          == args2.num_epochs,    'Please check training epoch(num_epoch)'
    assert args.gamma               == args2.gamma,         'please check gamma'
    # assert args.batch_size          == args2.batch_size,    'please check batch size'

    if hasattr(args, 'lr_drop_epoch') and lr_drop_revise == False:
        assert args.lr_drop_epoch   == args2.lr_drop_epoch, 'please check lr drop epoch'
    if hasattr(args, 'bottleneck'):
        assert args.bottleneck      == args2.bottleneck,    'please check bottleneck'

    if args.net_type == args2.net_type:
        assert args.lr                  == args2.lr,        'Please check learning rate: lr'




## ======================================================##
## ===================== Plot table =====================##
## ======================================================##


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
    #          rotation_mode="anchor")
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            kw.update(color='white')
            text = im.axes.text(j, i, valfmt(data[i, j], None), size=15, **kw)
            texts.append(text)

    return texts