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

    def __init__(self):
        super(ConvNet, self).__init__()

        self.name = self.__class__.__name__

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(F.dropout(self.conv2(x), training=self.training), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
