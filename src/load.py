import os
import numpy as np
from scipy.io import wavfile


# Read the wave file, and check its length (number of samples)
def load_data(args, class_name, file_name, check_length=True):
    file_path = os.path.join(args.data_root, 'dataset', class_name, file_name)
    if class_name == 'background_noise':
        file_path = os.path.join(args.data_root, 'background_noise', file_name)
    sampling_rate, signal = wavfile.read(file_path)
    file_dir = os.path.join(class_name, file_name)
    # Ensure that the sampling rate of the current file is correct
    assert sampling_rate == args.signal_sr, '{}'.format(file_dir)
    if check_length:
        # Ensure that the length of the current file is correct
        assert signal.shape[0] == args.signal_nsamples, '{}'.format(file_dir)
    return signal


# Read a random one-second-length segment from a random background noise file
def load_silence(args, noise_files_list, noise_probability_distribution):
    file_name = np.random.choice(noise_files_list, p=noise_probability_distribution)
    signal = load_data(args, 'background_noise', file_name, check_length=False)
    start_index = np.random.randint(0, signal.shape[0] - args.signal_nsamples)
    silence = signal[start_index:start_index + args.signal_nsamples]
    assert silence.shape[0] == args.signal_nsamples
    return silence
