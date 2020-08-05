import torch
from torch import nn
from torch.nn import functional as F


class LogisticRegression(nn.Module):

    def __init__(self, args):
        super(LogisticRegression, self).__init__()

        self.name = self.__class__.__name__

        self.input_shape = args.nmfcc * args.nfilter
        self.dropout = nn.Dropout(p=args.dropout);
        self.fc = nn.Linear(self.input_shape, args.nclass)

    def forward(self, x):
        x = x.view(-1, self.input_shape)
        x = self.dropout(x);
        x = self.fc(x)
        return x


class CompressModel(nn.Module):

    def __init__(self, args):
        super(CompressModel, self).__init__()

        self.name = self.__class__.__name__

        self.input_shape = args.nmfcc * args.nfilter
        self.dropout = nn.Dropout(p=args.dropout);
        self.fc1 = nn.Linear(self.input_shape, 1)
        self.fc2 = nn.Linear(1, args.nclass)

    def forward(self, x):
        x = x.view(-1, self.input_shape)
        x = self.dropout(x);
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
                     nn.Dropout2d(args.droupout)]
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


class ResNet(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.name = self.__class__.__name__;

        n_labels = args.nclass
        n_maps = args.nchannel
        self.conv0 = nn.Conv2d(1, n_maps, (3, 3), padding=(1, 1), bias=False)

        if args.res_pool != (1, 1):
            self.pool = nn.AvgPool2d(args.res_pool)

        self.n_layers = n_layers = args.nlayer;
        dilation = args.use_dilation

        if dilation:
            self.convs = [nn.Conv2d(n_maps, n_maps, (3, 3), padding=int(2**(i // 3)), dilation=int(2**(i // 3)),
                bias=False) for i in range(n_layers)]
        else:
            self.convs = [nn.Conv2d(n_maps, n_maps, (3, 3), padding=1, dilation=1,
                bias=False) for _ in range(n_layers)]

        for i, conv in enumerate(self.convs):
            self.add_module("bn{}".format(i + 1), nn.BatchNorm2d(n_maps, affine=False))
            self.add_module("conv{}".format(i + 1), conv)

        self.output = nn.Linear(n_maps, n_labels)

    def forward(self, x):

        for i in range(self.n_layers + 1):
            y = F.relu(getattr(self, "conv{}".format(i))(x))
            if i == 0:
                if hasattr(self, "pool"):
                    y = self.pool(y)
                old_x = y
            if i > 0 and i % 2 == 0:
                x = y + old_x
                old_x = x
            else:
                x = y
            if i > 0:
                x = getattr(self, "bn{}".format(i))(x)

        x = x.view(x.size(0), x.size(1), -1) # shape: (batch, feats, o3)
        x = torch.mean(x, 2)
        return self.output(x)

