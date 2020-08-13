import argparse
import os
import numpy as np
import random
import torch


def set_defaults():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # comet_ml configs
    parser.add_argument('--comet_key', type=str, default='')
    parser.add_argument('--comet_project', type=str, default='')
    parser.add_argument('--comet_workspace', type=str, default='')

    # data files configs
    parser.add_argument('--data_root', type=str, default='.', help='path to dataset')

    # audio data configs
    parser.add_argument('--signal_sr', type=int, default=16000, help='')
    parser.add_argument('--signal_len', type=float, default=1, help='')
    parser.add_argument('--nsilence', type=int, default=-1, help='')
    parser.add_argument('--silence_vol', type=float, default=0.5, help='')

    # augmentations LogFBEs | MFCCs
    parser.add_argument('--features_name', type=str, default='ta.LogFBEs',
                        help='LogFBEs | MFCCs | ta.MFCCs | ta.LogFBEs')
    parser.add_argument('--nfilt', type=int, default=40, help='')
    parser.add_argument('--numcep', type=int, default=13, help='')
    parser.add_argument('--winlen', type=float, default=0.025, help='')
    parser.add_argument('--winstep', type=float, default=0.010, help='')
    parser.add_argument('--nfft', type=int, default=512, help='')
    parser.add_argument('--preemph', type=float, default=0.97, help='')
    parser.add_argument('--ceplifter', type=int, default=22, help='')

    # augmentations 1
    parser.add_argument('--use_augmentations', action='store_true', default=False)
    parser.add_argument('--scale_min', type=float, default=0.95, help='')
    parser.add_argument('--scale_max', type=float, default=1.1, help='')
    parser.add_argument('--shift_min', type=float, default=-0.2, help='')
    parser.add_argument('--shift_max', type=float, default=0.2, help='')
    parser.add_argument('--noise_vol', type=float, default=0.5, help='')

    # augmentations 2
    parser.add_argument('--p_transform', type=float, default=0.2)
    parser.add_argument('--mask_time', type=int, default=12)
    parser.add_argument('--mask_freq', type=int, default=5)


    # experiment configs
    parser.add_argument('--name', type=str, default='untitled', help='name of the experiment')
    parser.add_argument('--outf', type=str, default='./output', help='folder to output images and model checkpoints')
    parser.add_argument('--seed', default=-1, type=int, help='manual seed')
    parser.add_argument('--frq_log', type=int, default=4, help='frequency of showing training results on console')
    parser.add_argument('--debug', type=int, default=32 * 4, help='')
    parser.add_argument('--test', action='store_true', default=False, help='load weights and run on test set')
    parser.add_argument('--weights_path', type=str, default=None, help='')

    # training configs
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--nepoch', type=int, default=6, help='number of epochs to train for')
    parser.add_argument('--batchsize', type=int, default=32, help='input batch size')
    parser.add_argument('--metric', type=str, default='acc', help='')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.0, help='')
    parser.add_argument('--beta1', type=float, default=0.9, help='')
    parser.add_argument('--optimizer', type=str, default='sgd', help='adam | sgd')
    parser.add_argument('--scheduler', type=str, default='none', help='auto | set | none')
    parser.add_argument('--scheduler_factor', type=float, default=0.5, help='')

    # model architecture
    parser.add_argument('--model', type=str, default='ConvNet',
                        help='LogisticRegression | CompressModel | ConvNet | ResNet | MatlabModel')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--nlayer', type=int, default=3)
    parser.add_argument('--nchannel', type=int, default=8)
    parser.add_argument('--res_pool', type=tuple, default=(1, 1))
    parser.add_argument('--use_dilation', action='store_true', default=False)

    args, unknown = parser.parse_known_args()
    # args = parser.parse_args();
    return args


def get_args():
    args = set_defaults()

    expr_dir = os.path.join(args.outf, args.name)
    if not os.path.isdir(expr_dir):
        os.makedirs(expr_dir)

    if args.seed != -1:
        print('using manual seed: {}'.format(args.seed))
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True

    return args
