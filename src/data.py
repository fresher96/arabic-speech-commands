import torch
import numpy as np
import os
from scipy.io import wavfile
import torchaudio


from src import utils
from src import transforms
from src.ClassDict import ClassDict
from src import load


class ASCDataset(torch.utils.data.Dataset):

    def __init__(self, data_root, dataset, transform, s_transform, nsilence, signal_samples, signal_sr, noise_pkg):
        super().__init__()

        self.transform = transform
        self.s_transform = s_transform
        self.data_root = data_root
        self.signal_samples = signal_samples
        self.signal_sr = signal_sr

        dataset = list(zip(*dataset))
        self.audio_files = list(dataset[0])
        self.audio_labels = dataset[1]
        self.audio_labels = [ClassDict.getId(name) for name in self.audio_labels]

        if nsilence == -1:
            nsilence = len(self.audio_files) // ClassDict.len()

        self.nsilence = nsilence
        self.silence_label = ClassDict.len()

        self.nfl, self.npd = utils.get_noise_files(noise_pkg, signal_sr)

    def load_silence(self):
        signal = load.load_silence(self.nfl, self.npd, self.signal_samples, self.data_root, self.signal_sr)
        tensor = self.s_transform(signal)
        return tensor

    def load_audio(self, idx):
        file_path = os.path.join(self.data_root, *self.audio_files[idx].split('\\'))
        # file_path = load.load_data(self.class_name, file_name, signal_samples, self.data_root, signal_sr)
        sampling_rate, signal = wavfile.read(file_path)
        tensor = self.transform(signal)
        return tensor

    def __getitem__(self, index):
        if index >= len(self.audio_labels):
            return self.load_silence(), self.silence_label
        else:
            return self.load_audio(index), self.audio_labels[index]

    def __len__(self):
        return len(self.audio_labels) + self.nsilence


def get_transform(args):

    def debug(tensor):
        print(tensor)
        return tensor

    args.signal_width = int(np.ceil((args.signal_len - args.winlen) / args.winstep) + 1)
    if args.features_name.lower() == 'logfbes':
        features = transforms.Compose([
            transforms.LogFBEs(args.signal_sr, args.winlen, args.winstep, args.nfilt,
                                    args.nfft, args.preemph),
            transforms.ToTensor(),
            ])
        args.nfeature = args.nfilt
    elif args.features_name.lower() == 'mfccs':
        features = transforms.Compose([
            transforms.MFCCs(args.signal_sr, args.winlen, args.winstep, args.numcep, args.nfilt,
                                    args.nfft, args.preemph, args.ceplifter),
            transforms.ToTensor(),
            ])
        args.nfeature = args.numcep
    elif args.features_name.lower() == 'ta.mfcc':
        features = transforms.Compose([
            transforms.ToTensor(),
            torchaudio.transforms.MFCC(sample_rate=args.signal_sr, n_mfcc=args.numcep),
        ]);
        args.nfeature = args.numcep
        args.signal_width = 81;
    else:
        raise Exception('--features_name should be one of {LogFBEs | MFCCs | ta.mfcc}')

    test_trasform = transforms.Compose([
        features,
        transforms.Lambda(lambda x: x.unsqueeze(0)),
    ])

    silence_transform = test_trasform

    args.signal_samples = args.signal_sr * args.signal_len
    args.bkg_noise_path = 'background_noise'

    noise_files, noise_probability_distribution = utils.get_noise_files(
        os.path.join(args.data_root, args.bkg_noise_path), signal_sr=args.signal_sr)

    train_transform = transforms.Compose([
        transforms.TimeScaling(scale_min=args.scale_min, scale_max=args.scale_max),
        transforms.TimeShifting(shift_min=args.shift_min, shift_max=args.shift_max),
        transforms.AddNoise(noise_files, noise_probability_distribution, args.noise_vol,
                            args.signal_samples, args.data_root, args.signal_sr),
        test_trasform,
    ])

    return {'train': train_transform, 'val': test_trasform, 'test': test_trasform}, silence_transform


def get_dataloader(args):
    args.nclass = ClassDict.len() + (args.nsilence != 0)
    splits = ['train', 'val', 'test']

    transform, s_transform = get_transform(args)

    dataset = utils.read_splits(args.data_root)

    if args.debug != -1:
        dataset = {split: dataset[split][:args.debug] for split in splits}

    dataset = {split: ASCDataset(args.data_root, dataset[split], transform[split], s_transform, args.nsilence,
                                 args.signal_samples, args.signal_sr, os.path.join(args.data_root, args.bkg_noise_path))
               for split in splits}

    dataloader = {split: torch.utils.data.DataLoader(dataset=dataset[split],
                                                     batch_size=args.batchsize,
                                                     shuffle=(split == 'train'))
                  for split in splits}

    return dataloader


