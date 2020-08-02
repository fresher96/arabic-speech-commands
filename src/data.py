import torch
import numpy as np;
import os;
from scipy.io import wavfile;
import torchaudio


from src import utils;
from src import transforms;
from src.ClassDict import ClassDict;


class ASCDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, transform, nsilence):
        super().__init__()

        self.transform = transform;

        # TODO: delete
        dataset = [(x, dataset[x]) for x in dataset];

        dataset = list(zip(*dataset))
        self.audio_files = list(dataset[0])
        self.audio_labels = dataset[1]
        self.audio_labels = [ClassDict.getId(name) for name in self.audio_labels];

        if(nsilence == -1):
            nsilence = len(self.audio_files) // len(set(self.audio_labels));

        self.nsilence = nsilence;
        self.silence_label = ClassDict.len();

    def load_silence(self):
        # TODO
        signal = np.random.randn(16000);
        transform = self.transform.transforms[-1];
        tensor = transform(signal);
        return tensor;

    def load_audio(self, path):
        # TODO
        # len = 16000;
        # signal = np.random.randn(len);

        (_, signal) = wavfile.read(path);

        tensor = self.transform(signal);
        return tensor;

    def __getitem__(self, index):
        if index >= len(self.audio_labels):
            return self.load_silence(), self.silence_label
        else:
            return self.load_audio(self.audio_files[index]), self.audio_labels[index]

    def __len__(self):
        return len(self.audio_labels) + self.nsilence


def get_transform(args):
    # transform = transforms.Compose([transforms.Resize(opt.isize),
    #                                 transforms.CenterCrop(opt.isize),
    #                                 transforms.ToTensor(),
    #                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

    test_trasform = transforms.Compose([
        torchaudio.transforms.MFCC(),
        transforms.ToTensor(),
    ]);

    train_transform = transforms.Compose([
        # transforms.TimeShift(args.left_shift, args.right_shift),
        # transforms.ToTensor(),
        test_trasform,
    ]);


    return {'train': train_transform, 'val': test_trasform, 'test': test_trasform};


def get_dataloader(args):
    splits = ['train', 'val', 'test']

    transform = get_transform(args);

    dataset = utils.split(args.dataroot, args.pct_val, args.pct_test);

    dataset = {split: ASCDataset(dataset[split], transform[split], args.nsilence)
               for split in splits}

    dataloader = {split: torch.utils.data.DataLoader(dataset=dataset[split],
                                                     batch_size=args.batchsize,
                                                     shuffle=(split == 'train'),
                                                     worker_init_fn=(None if args.seed == -1
                                                     else lambda _: np.random.seed(args.seed)))
                  for split in splits}

    return dataloader

"""
signal:         1D  args.signal_sr * args.signal_len
spectrogram:    2D  height = args.nmfcc * width = args.nfilter
tensor:         same
"""
