import torch
import numpy as np;
import os;
from scipy.io import wavfile;
import torchaudio


from src import utils;
from src import transforms;
from src.ClassDict import ClassDict;


class ASCDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, transform, s_transform, nsilence):
        super().__init__()

        self.transform = transform;
        self.s_transform = s_transform;

        dataset = list(zip(*dataset))
        self.audio_files = list(dataset[0])
        self.audio_labels = dataset[1]
        self.audio_labels = [ClassDict.getId(name) for name in self.audio_labels];

        if(nsilence == -1):
            nsilence = len(self.audio_files) // ClassDict.len();

        self.nsilence = nsilence;
        self.silence_label = ClassDict.len();

    def load_silence(self):
        # TODO
        signal = np.random.randn(16000);
        tensor = self.s_transform(signal);
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

    def debug(tensor):
        print(tensor);
        return tensor;

    test_trasform = transforms.Compose([
        # transforms.ToTensor(),
        # Lambda(debug),
        Lambda(lambda x: torch.from_numpy(x / 2**15).float()),
        # Lambda(debug),
        torchaudio.transforms.MFCC(n_mfcc=args.nmfcc),
        # Lambda(debug),
    ]);

    silence_transform = test_trasform;

    train_transform = transforms.Compose([
        # transforms.TimeShift(args.left_shift, args.right_shift),
        # transforms.ToTensor(),
        test_trasform,
    ]);


    return {'train': train_transform, 'val': test_trasform, 'test': test_trasform}, silence_transform;


def get_dataloader(args):
    args.nclass = ClassDict.len() + 1;
    splits = ['train', 'val', 'test']

    transform, s_transform = get_transform(args);

    dataset = utils.split(args.dataroot, args.pct_val, args.pct_test);
    # TODO: delete
    dataset = {split: [(x, dataset[split][x]) for x in dataset[split]] for split in splits};
    dataset['train'] = dataset['train'][:args.debug]


    dataset = {split: ASCDataset(dataset[split], transform[split], s_transform, args.nsilence)
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


class Lambda(object):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'

