import torch;
import numpy as np;

from src.utils import *;

class Random_Transform():
    def __init__(self, height, width):
        self.height = height;
        self.width = width;

    def __call__(self, signal):
        return np.random.randn(self.height, self.width);


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
        return torch.from_numpy(spectrogram).float();

    def __repr__(self):
        return self.__class__.__name__ + '()'


class TimeShift(object):

    def __init__(self, left, right):
        self.left = left;
        self.right = right;

    def __call__(self, signal):
        return signal

    def __repr__(self):
        return self.__class__.__name__ + '()'


class MFCC(object):

    def __init__(self, param1, param2):
        pass;

    def __call__(self, signal):
        spectrogram = signal;
        return spectrogram;

    def __repr__(self):
        return self.__class__.__name__ + '()'

