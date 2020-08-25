import numpy as np
import matplotlib.pyplot as plt

from configs import get_args
from load import load_data
from transforms import LogFBEs, MFCCs


def plot(logfbank_features, mfcc_features, num_frames):
    plt.figure(13, figsize=(13.5, 6.5))
    plt.subplots_adjust(left=0.08, right=1.1, bottom=0.08, top=0.95, hspace=0.75)

    plt.subplot(211)
    img = plt.imshow(logfbank_features.T, origin='lower', aspect='auto')
    plt.colorbar(img)
    plt.title('Log Filterbank Energies (LogFBEs)')
    plt.xlabel('Time (s)')
    plt.xlim(0, num_frames - 1)
    plt.xticks(np.linspace(0, num_frames - 1, 6), np.round(np.arange(0, 1.1, 0.2), decimals=1))
    plt.ylabel('Filter index')

    plt.subplot(212)
    img = plt.imshow(mfcc_features.T, origin='lower', aspect='auto')
    plt.colorbar(img)
    plt.title('Mel-Frequency Cepstral Coefficients (MFCCs)')
    plt.xlabel('Time (s)')
    plt.xlim(0, num_frames - 1)
    plt.xticks(np.linspace(0, num_frames - 1, 6), np.round(np.arange(0, 1.1, 0.2), decimals=1))
    plt.ylabel('Cepstrum index')
    y_ticks = np.arange(0, mfcc_features.shape[1] - 1, step=2)
    plt.yticks(y_ticks, y_ticks + 1)
    plt.savefig('../output/LogFBEs and MFCCs.png', bbox_inches='tight')


def main():
    args = get_args()

    class_name = 'rotate'
    file_name = '00000020_NO_07.wav'
    signal_samples = args.signal_len * args.signal_sr

    # Read a sample from the dataset for testing
    signal = load_data(class_name, file_name, signal_samples, args.data_root, args.signal_sr)

    logfbank = LogFBEs(samplerate=args.signal_sr, winlen=args.winlen, winstep=args.winstep,
                       nfilt=args.nfilt, nfft=args.nfft, preemph=args.preemph)
    logfbank_features = logfbank(signal)

    mfcc = MFCCs(samplerate=args.signal_sr, winlen=args.winlen, winstep=args.winstep, numcep=args.numcep,
                 nfilt=args.nfilt, nfft=args.nfft, preemph=args.preemph, ceplifter=args.ceplifter)
    mfcc_features = mfcc(signal)[:, 1:]

    num_frames = mfcc_features.shape[0]

    plot(logfbank_features, mfcc_features, num_frames)
    plt.show()


if __name__ == '__main__':
    main()
