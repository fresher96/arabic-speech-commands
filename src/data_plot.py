import os
import numpy as np
import matplotlib.pyplot as plt

from configs import get_args
import torchaudio


def plot(args, spectrogram, mel_spectrogram, mfcc, num_frames):
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

    plot(args, spectrogram, mel_spectrogram, mfcc, num_frames)
    plt.show()


if __name__ == '__main__':
    main()
