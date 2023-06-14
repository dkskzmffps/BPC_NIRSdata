import argparse
import sys
import JH_error_list

def net_param_setting(net_type, dataset):
    # =====================================================================================================================#
    # ==================================== Hyper Parameter of Network =====================================================#
    # =====================================================================================================================#

    depth = 101

    # depth for resnet, preact resnet, pyramid net
    # depth 20, 32, 44, 56, 110, 164, 1202: BasicBlock
    # 110, 164

    # net block type in the paper for data set 'image net'=================================================================#
    # depth 18, 34       : BasicBlock
    # depth 50, 101, 152 : Bottleneck

    if dataset == 'NIH_Covid':
        class_list = ['Loaded', 'Rapid', 'baseline']
    else:
        raise JH_error_list.Error_dataset_name

    parser = argparse.ArgumentParser(description='Network Parameter Setting')

    parser.add_argument('--net_type',   default=net_type,   type=str, help='model')
    parser.add_argument('--depth',      default=depth,      type=int, help='depth of model')
    parser.add_argument('--dataset',    default=dataset,    type=str, help='dataset info')
    parser.add_argument('--topk',       default=2,          type=int, help='')
    parser.add_argument('--class_list', default=class_list, type=list,help='class list')

    # Parameter for trainning
    lr              = 0.1               # initial learning rate
    # lr_drop_epoch   = [30, 60, 90]   # learning rate drop schedule,
    # num_epochs      = 1              # number of epochs
    lr_drop_epoch   = [30, 60, 90, 120]   # learning rate drop schedule,
    num_epochs      = 150               # number of epochs
    batch_size      = 32

    parser.add_argument('--num_epochs',      default=num_epochs,     type=int,   help='total number of train epoch')
    parser.add_argument('--lr',              default=lr,             type=float, help='learning_rate')
    parser.add_argument('--gamma',           default=0.1,            type=float, help='weight for learning_rate on lr drop epoch')
    parser.add_argument('--lr_drop_epoch',   default=lr_drop_epoch,  type=int,   help='step for learning rate down')
    parser.add_argument('--bottleneck',      default=True,                       help='choose bottleneck option')
    parser.add_argument('--start_epoch',    default=1,              type=int,   help='start_epoch')
    parser.add_argument('--batch_size',      default=batch_size,     type=int,   help='batch size')
    parser.add_argument('--test_batch_size', default=batch_size * 2, type=int, help='test_batch_size')


    parser.add_argument('--resume',     default=False, help='resume from checkpoint')
    parser.add_argument('--testOnly',   default=False, help='Test mode with the saved model')
    # parser.add_argument('--save_class_score',       default=False,                      help='Save class score with the saved model')

    return parser.parse_args()