import torch
from torch import nn
from torch.nn import functional as F


class LogisticRegression(nn.Module):

    def __init__(self, args):
        super(LogisticRegression, self).__init__()

        self.name = self.__class__.__name__

        self.input_shape = args.nmfcc * args.nfilter
        self.fc = nn.Linear(self.input_shape, args.nclass)

    def forward(self, x):
        x = x.view(-1, self.input_shape)
        x = self.fc(x)
        return x


class CompressModel(nn.Module):

    def __init__(self, args):
        super(CompressModel, self).__init__()

        self.name = self.__class__.__name__

        self.input_shape = args.nmfcc * args.nfilter
        self.fc1 = nn.Linear(self.input_shape, 1)
        self.fc2 = nn.Linear(1, args.nclass)

    def forward(self, x):
        x = x.view(-1, self.input_shape)
        x = F.dropout(x, p=0.0, training=self.training);
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class ConvNet(nn.Module):

    def __init__(self, args):
        super(ConvNet, self).__init__()

        self.name = self.__class__.__name__

        def block(in_filters, out_filters, bn):
            block = [nn.Conv2d(in_filters, out_filters, 3, bias=not bn),
                     nn.ReLU(0.2),
                     nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters))
            return nn.Sequential(*block);

        self.conv = nn.Sequential();
        for i in range(args.nlayer):
            self.conv.add_module('conv_%d'%i, block(2**i, 2**(i+1), i != 0));

        self.fc = nn.Linear(2 ** args.nlayer, args.nclass);

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), x.size(1), -1)  # shape: (batch, feats, o3)
        x = torch.mean(x, 2)
        x = self.fc(x)
        return x
