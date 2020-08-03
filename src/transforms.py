import torch
import numpy as np

from src.utils import *
from python_speech_features import mfcc, logfbank


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

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __call__(self, signal):
        return signal

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

