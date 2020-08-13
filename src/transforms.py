import torch
import numpy as np
import random
import torchaudio

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
        bkg_noise_volume = random.uniform(0.0, self.volume_rate)
        return signal + np.array(bkg_noise * bkg_noise_volume, dtype=np.int16)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class TimeShifting(object):

    def __init__(self, shift_min, shift_max):
        self.shift_min = shift_min
        self.shift_max = shift_max

    def __call__(self, signal):
        shift_rand = random.uniform(self.shift_min, self.shift_max)
        num_pad = int(shift_rand * len(signal))
        padding = np.zeros(abs(num_pad), dtype=np.int16)
        if num_pad > 0:
            return np.concatenate((padding, signal[:-num_pad]))
        elif num_pad < 0:
            return np.concatenate((signal[-num_pad:], padding))
        return signal

    def __repr__(self):
        return self.__class__.__name__ + '()'


class TimeScaling(object):

    def __init__(self, scale_min, scale_max):
        self.scale_min = scale_min
        self.scale_max = scale_max

    def __call__(self, signal):
        rate = random.uniform(self.scale_min, self.scale_max)
        signal = (signal / (2 ** 15)).astype(np.float32)
        stretched_signal = time_stretch(signal, rate)
        stretched_signal = (stretched_signal * (2 ** 15)).astype(np.int16)
        new_length = len(stretched_signal)

        if new_length < len(signal):
            padding = np.zeros(len(signal) - new_length, dtype=np.int16)
            index = np.random.randint(0, len(padding) + 1)
            return np.concatenate((padding[:index], stretched_signal, padding[index:]))
        index = np.random.randint(0, new_length - len(signal) + 1)
        return stretched_signal[index:index + len(signal)]

    def __repr__(self):
        return self.__class__.__name__ + '()'


class LogFBEs(object):

    def __init__(self, samplerate, winlen, winstep, nfilt, nfft, preemph):
        self.samplerate = samplerate
        self.winlen = winlen
        self.winstep = winstep
        self.nfilt = nfilt
        self.nfft = nfft
        self.preemph = preemph

    def __call__(self, signal):
        features = logfbank(signal, samplerate=self.samplerate, winlen=self.winlen,
                        winstep=self.winstep, nfilt=self.nfilt, nfft=self.nfft,
                        preemph=self.preemph)
        return features

    def __repr__(self):
        return self.__class__.__name__ + '()'


class MFCCs(object):

    def __init__(self, samplerate, winlen, winstep, numcep, nfilt, nfft, preemph, ceplifter):
        self.samplerate = samplerate
        self.winlen = winlen
        self.winstep = winstep
        self.numcep = numcep
        self.nfilt = nfilt
        self.nfft = nfft
        self.preemph = preemph
        self.ceplifter = ceplifter

    def __call__(self, signal):
        features = mfcc(signal, samplerate=self.samplerate, winlen=self.winlen,
                        winstep=self.winstep, numcep=self.numcep, nfilt=self.nfilt, nfft=self.nfft,
                        preemph=self.preemph, ceplifter=self.ceplifter)
        return features

    def __repr__(self):
        return self.__class__.__name__ + '()'



class TimeShifting2():

    def __init__(self, shift_min, shift_max):
        self.shift_min = shift_min
        self.shift_max = shift_max

    def __call__(self, signal):
        shift_rand = random.uniform(self.shift_min, self.shift_max)
        num_pad = int(shift_rand * len(signal))
        padding = torch.zeros(abs(num_pad))
        if num_pad > 0:
            return torch.cat((padding, signal[:-num_pad]))
        elif num_pad < 0:
            return torch.cat((signal[-num_pad:], padding))
        return signal


class AddNoise2():

    def __init__(self, noise_pkg, noise_vol, signal_samples, signal_sr):
        self.noise_pkg = noise_pkg
        self.nfl, self.npd = get_noise_files(noise_pkg, signal_sr)
        self.noise_vol = noise_vol
        self.signal_samples = signal_samples

    def load_path(self, path):
        x, sr = torchaudio.load(path)
        x = x.squeeze()
        return x

    def __call__(self, old_signal):
        file_name = np.random.choice(self.nfl, p=self.npd)
        file_path = os.path.join(self.noise_pkg, file_name)
        signal = self.load_path(file_path)
        start_index = np.random.randint(0, signal.size()[0] - self.signal_samples)
        silence = signal[start_index : start_index + self.signal_samples]

        return old_signal + silence * self.noise_vol


class RandomApplyTransform():
    def __init__(self, p, transform):
        self.p = p
        self.transform = transform

    def __call__(self, signal):
        if(random.random() < self.p):
            signal = self.transform(signal);
        return signal;
