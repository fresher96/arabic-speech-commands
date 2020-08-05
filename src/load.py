import os
import numpy as np
from scipy.io import wavfile


# Read the wave file, and check its length (number of samples)
def load_data(class_name, file_name, signal_samples, data_root, signal_sr, check_length=True):
    file_path = os.path.join(data_root, 'dataset', class_name, file_name)
    if class_name == 'background_noise':
        file_path = os.path.join(data_root, 'background_noise', file_name)
    sampling_rate, signal = wavfile.read(file_path)
    if signal_sr is not None:
        file_dir = os.path.join(class_name, file_name)
        # Ensure that the sampling rate of the current file is correct
        assert sampling_rate == signal_sr, '{}'.format(file_dir)
        if check_length:
            # Ensure that the length of the current file is correct
            assert signal.shape[0] == signal_samples, '{}'.format(file_dir)
    return signal


# Read a random one-second-length segment from a random background noise file
def load_silence(noise_files_list, noise_probability_distribution, signal_samples, data_root, signal_sr=None):
    file_name = np.random.choice(noise_files_list, p=noise_probability_distribution)
    signal = load_data('background_noise', file_name, signal_samples, data_root, signal_sr, check_length=False)
    start_index = np.random.randint(0, signal.shape[0] - signal_samples)
    silence = signal[start_index:start_index + signal_samples]
    return silence
