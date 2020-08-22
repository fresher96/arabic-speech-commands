import torch
from torch import nn
from torch.nn import functional as F


class LogisticRegression(nn.Module):

    def __init__(self, args):
        super(LogisticRegression, self).__init__()

        self.name = self.__class__.__name__

        self.input_shape = args.nfeature * args.signal_width
        self.dropout = nn.Dropout(p=args.dropout);
        self.fc = nn.Linear(self.input_shape, args.nclass)

    def forward(self, x):
        # print(x.size());
        x = x.view(-1, self.input_shape)
        x = self.dropout(x);
        x = self.fc(x)
        return x


class CompressModel(nn.Module):

    def __init__(self, args):
        super(CompressModel, self).__init__()

        self.name = self.__class__.__name__

        self.input_shape = args.nfeature * args.signal_width
        self.dropout = nn.Dropout(p=args.dropout);
        self.fc1 = nn.Linear(self.input_shape, 1, bias=False)
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
            block = [nn.Conv2d(in_filters, out_filters, 3, bias=not bn)]
            if bn: block.append(nn.BatchNorm2d(out_filters))
            block += [nn.ReLU()];
            return nn.Sequential(*block);

        init_fm = args.nchannel;
        self.conv = nn.Sequential();
        self.conv.add_module('conv_%d' % 0, block(1, init_fm, False));

        for i in range(1, args.nlayer):
            i -= 1
            self.conv.add_module('conv_%d'%(i+1), block(2**i * init_fm, 2**(i+1) * init_fm, True));

        self.fc = nn.Sequential(
            # nn.Linear(2 ** (args.nlayer - 1) * init_fm, 10, bias=False),
            # nn.Linear(10, args.nclass),
            nn.Dropout(p=args.dropout),
            nn.Linear(2 ** (args.nlayer - 1) * init_fm, args.nclass),
        );

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



class MatlabModel(nn.Module):

    """
    Taken from:
    https://www.mathworks.com/help/deeplearning/ug/deep-learning-speech-recognition.html

    layers = [
        imageInputLayer([numHops numBands])

        convolution2dLayer(3,numF,'Padding','same')
        batchNormalizationLayer
        reluLayer

        maxPooling2dLayer(3,'Stride',2,'Padding','same')

        convolution2dLayer(3,2*numF,'Padding','same')
        batchNormalizationLayer
        reluLayer

        maxPooling2dLayer(3,'Stride',2,'Padding','same')

        convolution2dLayer(3,4*numF,'Padding','same')
        batchNormalizationLayer
        reluLayer

        maxPooling2dLayer(3,'Stride',2,'Padding','same')

        convolution2dLayer(3,4*numF,'Padding','same')
        batchNormalizationLayer
        reluLayer
        convolution2dLayer(3,4*numF,'Padding','same')
        batchNormalizationLayer
        reluLayer

        maxPooling2dLayer([timePoolSize,1])

        dropoutLayer(dropoutProb)
        fullyConnectedLayer(numClasses)
        softmaxLayer
        weightedClassificationLayer(classWeights)
    ];

    !python main.py \
        --comet_key 'bLjz3xx3gKDZwM7Hm0Kcgbpww' --comet_project 'arabic-commands' --comet_workspace 'fresher96' \
        --data_root ../dataroot \
        --debug -1 \
        \
        --features_name ta.mfccs \
        --nfilt 128 \
        --numcep 64 \
        \
        --weight_decay 1e-3 \
        --nepoch 100 \
        --batchsize 128 \
        --lr 3e-3 \
        --optimizer adam \
        --scheduler auto \
        \
        --model MatlabModel \
        --dropout 0.20 \
        \
        --p_transform 0.1 \
        --mask_time 12 \
        --mask_freq 8 \
    """

    def __init__(self, args):
        super().__init__()
        self.name = self.__class__.__name__;

        numF = args.nchannel; # change to 40 or 41?

        w = args.signal_width;
        w = (w - 1) // 2 + 1;
        w = (w - 1) // 2 + 1;
        w = (w - 1) // 2 + 1;

        h = args.nfeature;
        h = (h - 1) // 2 + 1;
        h = (h - 1) // 2 + 1;
        h = (h - 1) // 2 + 1;

        s = 2;
        p = 1;

        # input dims: batchsize | channels = 1 | height = args.nfeature | width = args.signal_width
        self.layers = nn.Sequential(

            nn.Conv2d(in_channels=1, out_channels=numF, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=numF),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=s, padding=p),


            nn.Conv2d(in_channels=numF, out_channels=2*numF, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=2*numF),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=s, padding=p),


            nn.Conv2d(in_channels=2*numF, out_channels=4*numF, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=4*numF),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=s, padding=p),


            nn.Conv2d(in_channels=4 * numF, out_channels=4 * numF, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=4 * numF),
            nn.ReLU(),
            nn.Conv2d(in_channels=4 * numF, out_channels=4 * numF, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=4 * numF),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, w)),
            nn.Dropout2d(p=args.dropout),


            nn.Flatten(start_dim=1),
            # nn.Linear(in_features=4 * numF * h * 1, out_features=10, bias=False),
            # nn.Linear(in_features=10, out_features=args.nclass),
            nn.Linear(in_features=4 * numF * h * 1, out_features=args.nclass),
        );

    def forward(self, x):
        return self.layers(x);
