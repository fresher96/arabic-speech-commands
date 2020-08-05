import torch
import numpy as np
import random

from src.utils import *
from python_speech_features import mfcc, logfbank
from src.load import load_silence
from librosa.effects import time_stretch


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

class ToTensor(object):

    def __call__(self, spectrogram):
        return torch.from_numpy(spectrogram).float()

    def __repr__(self):
        return self.__class__.__name__ + '()'


class AddNoise(object):

    def __init__(self, noise_files, noise_probability_distribution, volume_rate, signal_samples, data_root, signal_sr):
        self.noise_files = noise_files
        self.noise_probability_distribution = noise_probability_distribution
        self.volume_rate = volume_rate
        self.signal_samples = signal_samples
        self.data_root = data_root
        self.signal_sr = signal_sr

    def __call__(self, signal):
        bkg_noise = load_silence(self.noise_files, self.noise_probability_distribution,
                                 self.signal_samples, self.data_root, self.signal_sr)
        bkg_noise_volume = random.uniform(0.01, self.volume_rate)
        return signal + np.array(bkg_noise * bkg_noise_volume, dtype=np.int16)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class TimeShifting(object):

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


class TimeScaling(object):

    def __init__(self, scale_min=0.1, scale_max=0.2, amp_min=-10, amp_max=10):
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.amp_min = amp_min
        self.amp_max = amp_max

    def __call__(self, signal):
        stretch = random.uniform(self.scale_min, self.scale_max)
        rate = 1 + random.choice([-1, 1]) * stretch
        signal = (signal / (2 ** 15)).astype(np.float32)
        stretched_signal = time_stretch(signal, rate)
        stretched_signal = (stretched_signal * (2 ** 15)).astype(np.int16)
        new_length = len(stretched_signal)

        if new_length < len(signal):
            padding = np.random.randint(self.amp_min, self.amp_max, len(signal) - new_length, dtype=np.int16)
            index = np.random.randint(0, len(padding) + 1)
            return np.concatenate((padding[:index], stretched_signal, padding[index:]))
        index = np.random.randint(0, new_length - len(signal) + 1)
        return stretched_signal[index:index + len(signal)]

    def __repr__(self):
        return self.__class__.__name__ + '()'


class LogFBEs(object):

    def __init__(self, samplerate, winlen, winstep, nfilt, nfft, preemph):
        self.samplerate = samplerate;
        self.winlen = winlen;
        self.winstep = winstep;
        self.nfilt = nfilt;
        self.nfft = nfft;
        self.preemph = preemph;

    def __call__(self, signal):
        features = logfbank(signal, samplerate=self.samplerate, winlen=self.winlen,
                        winstep=self.winstep, nfilt=self.nfilt, nfft=self.nfft,
                        preemph=self.preemph)
        return features

    def __repr__(self):
        return self.__class__.__name__ + '()'


class MFCCs(object):

    def __init__(self, samplerate, winlen, winstep, numcep, nfilt, nfft, preemph, ceplifter):
        self.samplerate = samplerate;
        self.winlen = winlen;
        self.winstep = winstep;
        self.numcep = numcep,
        self.nfilt = nfilt;
        self.nfft = nfft;
        self.preemph = preemph;
        self.ceplifter = ceplifter;

    def __call__(self, signal):
        features = mfcc(signal, samplerate=self.samplerate, winlen=self.winlen,
                        winstep=self.winstep, numcep=self.numcep, nfilt=self.nfilt, nfft=self.nfft,
                        preemph=self.preemph, ceplifter=self.ceplifter)
        return features

    def __repr__(self):
        return self.__class__.__name__ + '()'

