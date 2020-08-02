import argparse
import os
import numpy as np
import random
import torch

def set_defaults():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--signal_sr', type=int, default=16000, help='')
    parser.add_argument('--signal_len', type=float, default=1, help='')
    parser.add_argument('--nmfcc', type=int, default=40, help='')
    parser.add_argument('--nfilter', type=int, default=99, help='')
    parser.add_argument('--nsilence', type=int, default=-1, help='')


    parser.add_argument('--model', type=str, default='vanilla', help='chooses which model to use. ganomaly')
    parser.add_argument('--nclass', type=int, default=41, help='')
    parser.add_argument('--momentum', type=int, default=40, help='')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--epochs', type=int, default=15, help='number of epochs to train for')


    parser.add_argument('--dataroot', default=os.path.join('Arabic_Speech_Commands_Dataset', 'speech_commands'), help='path to dataset')
    parser.add_argument('--pct_val', type=float, default=0.20, help='')
    parser.add_argument('--pct_test', type=float, default=0.20, help='')

    parser.add_argument('--batchsize', type=int, default=64, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--name', type=str, default='untitled', help='name of the experiment')
    parser.add_argument('--outf', default='./output', help='folder to output images and model checkpoints')
    parser.add_argument('--seed', default=-1, type=int, help='manual seed')
    parser.add_argument('--frq_log', type=int, default=10, help='frequency of showing training results on console')
    parser.add_argument('--test', action='store_true', default=False, help='load weights and run on test set')

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
        print(args.seed);
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True


    return args
