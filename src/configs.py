import argparse
import os
import numpy as np
import random
import torch

def set_defaults():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # audio data configs
    parser.add_argument('--signal_sr', type=int, default=16000, help='')
    parser.add_argument('--signal_len', type=float, default=1, help='')
    parser.add_argument('--nmfcc', type=int, default=40, help='')
    parser.add_argument('--nfilter', type=int, default=81, help='')
    parser.add_argument('--nsilence', type=int, default=-1, help='')

    # training configs
    parser.add_argument('--model', type=str, default='vanilla', help='')
    parser.add_argument('--nepoch', type=int, default=2, help='number of epochs to train for')
    parser.add_argument('--batchsize', type=int, default=64, help='input batch size')
    parser.add_argument('--metric', type=str, default='loss', help='')
    parser.add_argument('--momentum', type=int, default=0.9, help='')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='')
    parser.add_argument('--optimizer', type=str, default='sgd', help='adam | sgd')

    # data files configs
    parser.add_argument('--dataroot', default=os.path.join('Arabic_Speech_Commands_Dataset', 'speech_commands'), help='path to dataset')
    parser.add_argument('--pct_val', type=float, default=0.20, help='')
    parser.add_argument('--pct_test', type=float, default=0.20, help='')

    # experiment configs
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--name', type=str, default='untitled', help='name of the experiment')
    parser.add_argument('--outf', default='./output', help='folder to output images and model checkpoints')
    parser.add_argument('--seed', default=8, type=int, help='manual seed')
    parser.add_argument('--frq_log', type=int, default=25, help='frequency of showing training results on console')
    parser.add_argument('--test', action='store_true', default=False, help='load weights and run on test set')
    parser.add_argument('--debug', type=int, default=-1, help='')

    # comet_ml configs
    parser.add_argument('--comet_key', type=str, default='bLjz3xx3gKDZwM7Hm0Kcgbpww', help='');
    parser.add_argument('--comet_project', type=str, default='arabic-commands', help='');
    parser.add_argument('--comet_workspace', type=str, default='fresher96', help='');

    return parser.parse_args();


def get_args():
    args = set_defaults();

    expr_dir = os.path.join(args.outf, args.name)
    if not os.path.isdir(expr_dir):
        os.makedirs(expr_dir)

    file_name = os.path.join(expr_dir, 'args.txt')
    with open(file_name, 'wt') as f:
        f.write('-------------- COMMAND LINE ARGUMENTS --------------\n')
        for k, v in sorted(vars(args).items()):
            f.write('%s: %s\n' % (str(k), str(v)))
        f.write('-------------- END --------------\n')

    if (args.seed != -1):
        print('using manual seed: {}'.format(args.seed));
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True


    return args
