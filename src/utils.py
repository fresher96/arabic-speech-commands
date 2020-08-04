import os
from math import ceil
import numpy as np
from scipy.io import wavfile
from collections import defaultdict
import pandas as pd


# Ensure that the current file is a wave file
def is_wav_file(file_name):
    return file_name.lower().endswith('.wav')


# Create a dictionary to match each class_name with its files
def get_dataset_files(dataset_path):
    # Create a dictionary of lists
    dataset_files = defaultdict(list)
    for dir_path, _, file_names in os.walk(dataset_path):
        for file_name in file_names:
            if is_wav_file(file_name):
                class_name = os.path.basename(dir_path)
                dataset_files[class_name].append(file_name)
    return dataset_files


# Create a list that contains all background noise files and their probability_distribution
def get_noise_files(bkg_noise_path, signal_sr=16000):
    files_list = list()
    for file_name in os.listdir(bkg_noise_path):
        if is_wav_file(file_name):
            files_list.append(file_name)
    signals_length = list()
    for file_name in files_list:
        file_path = os.path.join(bkg_noise_path, file_name)
        sampling_rate, signal = wavfile.read(file_path)
        file_dir = os.path.join('background_noise', file_name)
        # Ensure that the sampling rate of the current file is correct
        assert sampling_rate == signal_sr, '{}'.format(file_dir)
        signals_length.append(signal.shape[0])
    signals_length = np.array(signals_length)
    probability_distribution = signals_length / signals_length.sum()
    return files_list,  probability_distribution


# Group the list of files by person
def group_by_person(files_list):
    person = defaultdict(list)
    for file_name in files_list:
        person[file_name[:8]].append(file_name)
    return list(person.values())


# Dataset splitting: training set, validation set, test set
def split(args, validation_part=0.2, test_part=0.2):
    dataset_path = os.path.join(args.data_root, 'dataset')
    dataset_files = get_dataset_files(dataset_path)
    training_files, validation_files, test_files = [], [], []
    for class_name, files_list in dataset_files.items():
        files_lists = group_by_person(files_list)
        num_test = ceil(test_part * len(files_lists))
        num_validation = ceil(validation_part * len(files_lists))
        for i in range(num_test):
            for file_name in files_lists[i]:
                file_path = os.path.join('dataset', class_name, file_name)
                test_files.append((file_path, class_name))
        for i in range(num_test, num_test + num_validation):
            for file_name in files_lists[i]:
                file_path = os.path.join('dataset', class_name, file_name)
                validation_files.append((file_path, class_name))
        for i in range(num_test + num_validation, len(files_lists)):
            for file_name in files_lists[i]:
                file_path = os.path.join('dataset', class_name, file_name)
                training_files.append((file_path, class_name))
    return {'train': training_files, 'val': validation_files, 'test': test_files}


def split_to_csv(args, dataset_splits):
    d = defaultdict(lambda: defaultdict(list))
    for set_name, set_data in dataset_splits.items():
        for file_path, class_name in set_data:
            d[set_name]['file'].append(file_path)
            d[set_name]['class'].append(class_name)

    for set_name, set_dict in d.items():
        data_frame = pd.DataFrame(set_dict)
        data_frame.to_csv(set_name + '_' + args.features_name + '.csv', index=False)
