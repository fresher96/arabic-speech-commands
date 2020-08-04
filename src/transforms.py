import torch
import numpy as np
import random

from src.utils import *
from python_speech_features import mfcc, logfbank
from src.load import load_silence


class Random_Transform(object):

    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, signal):
        return np.random.randn(self.height, self.width)


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor(object):

    def __call__(self, spectrogram):
        return torch.from_numpy(spectrogram).float()

    def __repr__(self):
        return self.__class__.__name__ + '()'


class TimeShift(object):

    def __init__(self, sec_min=0.1, sec_max=0.3, amp_min=-10, amp_max=10):
        self.sec_min = sec_min
        self.sec_max = sec_max
        self.amp_min = amp_min
        self.amp_max = amp_max

    def __call__(self, signal):
        sec = random.uniform(self.sec_min, self.sec_max)
        num_pad = int(sec * len(signal))
        padding = np.random.randint(self.amp_min, self.amp_max, num_pad, dtype=np.int16)
        if random.choice([0, 1]) == 0:
            return np.concatenate((padding, signal[:-num_pad]))
        return np.concatenate((signal[num_pad:], padding))

    def __repr__(self):
        return self.__class__.__name__ + '()'


class LogFBEs(object):

    def __init__(self, n_filters):
        self.n_filters = n_filters

    def __call__(self, signal):
        features = logfbank(signal, nfilt=self.n_filters)
        return features

    def __repr__(self):
        return self.__class__.__name__ + '()'


class MFCCs(object):

    def __init__(self, n_filters):
        self.n_filters = n_filters

    def __call__(self, signal):
        features = mfcc(signal, nfilt=self.n_filters)
        return features

    def __repr__(self):
        return self.__class__.__name__ + '()'

