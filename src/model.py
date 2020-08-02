import torch;
from torch import nn;
from torch.nn import functional as F


class LogisticRegression(nn.Module):

    def __init__(self, args):
        super(LogisticRegression, self).__init__()

        self.name = "LogisticRegression";

        self.input_shape = args.nmfcc * args.nfilter;
        self.fc = nn.Linear(self.input_shape, args.nclass);

    def forward(self, x):
        x = x.view(-1, self.input_shape)
        x = self.fc(x);
        return x
