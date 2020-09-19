import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import json

from configs import get_args
import torchaudio


def plot_features(args, spectrogram, mel_spectrogram, mfcc, num_frames):
    plt.figure(1, figsize=(8, 8))
    plt.subplots_adjust(left=0.08, right=1.1, bottom=0.08, top=0.95, hspace=0.75)

    plt.subplot(311)
    img = plt.imshow(np.log(spectrogram[0, :, :].numpy()), origin='lower', aspect='auto')
    plt.colorbar(img)
    plt.title(r'$\bf{(a) \ Spectrogram}$')
    plt.xlabel('Time (s)')
    plt.xlim(0, num_frames - 1)
    plt.xticks(np.linspace(0, num_frames - 1, 6), np.round(np.arange(0, 1.1, 0.2), decimals=1))
    plt.ylabel('Frequency (KHz)')
    y_ticks = np.linspace(0, args.nfft / 2 + 1, (int(args.signal_sr / 2000) + 1))
    plt.yticks(y_ticks, (np.arange(0, int(args.signal_sr / 2) + 1, step=1000) / 1000).astype(int))

    plt.subplot(312)
    img = plt.imshow(np.log(mel_spectrogram[0, :, :].numpy()), origin='lower', aspect='auto')
    plt.colorbar(img)
    plt.title(r'$\bf{(b) \ LogFBEs}$')
    plt.xlabel('Time (s)')
    plt.xlim(0, num_frames - 1)
    plt.xticks(np.linspace(0, num_frames - 1, 6), np.round(np.arange(0, 1.1, 0.2), decimals=1))
    plt.ylabel('Filter index')

    plt.subplot(313)
    img = plt.imshow(mfcc[0, 1:, :].numpy(), origin='lower', aspect='auto')
    plt.colorbar(img)
    plt.title(r'$\bf{(c) \ MFCCs}$')
    plt.xlabel('Time (s)')
    plt.xlim(0, num_frames - 1)
    plt.xticks(np.linspace(0, num_frames - 1, 6), np.round(np.arange(0, 1.1, 0.2), decimals=1))
    plt.ylabel('Cepstrum index')
    y_ticks = np.arange(0, mfcc.size()[1] - 1, step=2)
    plt.yticks(y_ticks, y_ticks + 1)
    plt.savefig('../output/Spectrogram, LogFBEs, and MFCCs.png', bbox_inches='tight')


def plot_history(args, history):
    # Plot training and validation accuracy values
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))
    fig.subplots_adjust(hspace=0.5)
    legend = ['Training', 'Validation']

    axes[0].plot(np.array(history[0]["y"]), color='blue')
    axes[0].plot(np.array(history[2]["y"]), color='black')
    axes[0].set_title('Accuracy (%)')
    axes[0].set_xlabel('Epoch')
    axes[0].legend(legend, loc='lower right')
    axes[0].xaxis.set_major_locator(MultipleLocator(base=5))
    axes[0].yaxis.set_major_locator(MultipleLocator(base=10))
    # Add the grid
    axes[0].grid(which='major', axis='both', linestyle='dotted')

    # Plot training and validation loss values
    axes[1].plot(history[1]["y"], color='red')
    axes[1].plot(history[3]["y"], color='black')
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend(legend, loc='upper right')
    axes[1].xaxis.set_major_locator(MultipleLocator(base=5))
    axes[1].yaxis.set_major_locator(MultipleLocator(base=0.5))
    # Add the grid
    axes[1].grid(which='major', axis='both', linestyle='dotted')

    plt.savefig('../output/History.png', bbox_inches='tight')


def main():
    args = get_args()

    class_name = 'rotate'
    file_name = '00000020_NO_07.wav'
    signal_path = os.path.join(args.data_root, 'dataset', class_name, file_name)

    # # Read a sample from the dataset for testing
    # signal = load_data(class_name, file_name, signal_samples, args.data_root, args.signal_sr)

    signal, _ = torchaudio.load(signal_path)

    melkwargs = {
        'n_mels': args.nfilt,
        'n_fft': args.nfft,
        'win_length': int(args.winlen * args.signal_sr),
        'hop_length': int(args.winstep * args.signal_sr)
    }

    spectrogram = torchaudio.transforms.Spectrogram(n_fft=args.nfft,
                                                    win_length=int(args.winlen * args.signal_sr),
                                                    hop_length=int(args.winstep * args.signal_sr))(signal)

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=args.signal_sr,
                                                           n_fft=args.nfft,
                                                           n_mels=args.nfilt,
                                                           win_length=int(args.winlen * args.signal_sr),
                                                           hop_length=int(args.winstep * args.signal_sr))(signal)

    mfcc = torchaudio.transforms.MFCC(sample_rate=args.signal_sr,
                                      n_mfcc=args.numcep,
                                      log_mels=True,
                                      melkwargs=melkwargs)(signal)

    num_frames = spectrogram.size()[2]

    plot_features(args, spectrogram, mel_spectrogram, mfcc, num_frames)

    history_path = os.path.join('..', 'assets', 'history.json')

    # Load the history file
    with open(history_path, mode='rb') as history_file:
        history = json.load(history_file)

    plot_history(args, history)

    plt.show()


if __name__ == '__main__':
    main()
