import os
import numpy as np
import pyaudio
from threading import Timer
# from keras.models import load_model
from tensorflow.keras.models import load_model

from src.configs import get_args
from src.ClassDict import ClassDict
from src.transforms import MFCCs, LogFBEs


class StreamingAudio:

    def __init__(self, args):
        self.args = args

        model_name = self.args.model + '_' + self.args.features_name
        model_path = os.path.join('..', 'output', 'models', model_name + '.model')

        self.model = load_model(model_path)

        self.signal_samples = self.args.signal_len * self.args.signal_sr

        # Create an array of length that equals to signal_samples, and initialize it with zeros
        self.signal = np.zeros(self.signal_samples)

        self.silence_idx = ClassDict.len()
        # Create an array that contains the last num_chunks predictions (updated continuously)
        self.predictions = np.full(self.args.n_chunks, self.silence_idx)

        self.chunk_size = int((self.args.signal_len / self.args.n_chunks) * self.signal_samples)

        # Identify the properties of the streaming data
        self.stream = pyaudio.PyAudio().open(
            rate=self.args.signal_sr,
            # Number of input channels
            channels=1,
            # The type of streaming audio data
            format=pyaudio.paInt16,
            input=True,
            output=True,
            frames_per_buffer=self.chunk_size
        )

        self.mfccs = MFCCs(self.args.signal_sr, self.args.winlen, self.args.winstep, self.args.numcep,
                           self.args.nfilt, self.args.nfft, self.args.preemph, self.args.ceplifter)

        self.logfbes = LogFBEs(self.args.signal_sr, self.args.winlen, self.args.winstep,
                               self.args.nfilt, self.args.nfft, self.args.preemph)

    def start(self):
        # Read a chunk from the audio stream
        chunk_data = self.stream.read(self.chunk_size)
        # Convert the chunk of data to int16
        chunk_signal = np.fromstring(chunk_data, dtype=np.int16)
        # Discard the oldest chunk and append the new one
        self.signal = np.concatenate((self.signal[self.chunk_size:], chunk_signal))

        # Extract features from the current signal
        x = self.extract_features()

        # Reshape the sample according to the chosen model
        if self.args.model.lower() == 'cnn':
            x = x.reshape((1, x.shape[0], x.shape[1], 1))

        # Predict the received speech command
        prediction = self.model.predict(x)[0]

        res_idx = self.silence_idx if prediction.max() < self.args.p_threshold else prediction.argmax()
        self.predictions = np.concatenate((self.predictions[1:], np.array(res_idx).reshape(1)))
        (values, counts) = np.unique(self.predictions, return_counts=True)
        command_idx = int(values[np.argmax(counts)])

        predicted_command = 'silence' if command_idx == self.silence_idx else ClassDict.getName(command_idx)

        print(predicted_command)

        timer = Timer(self.args.signal_len / self.args.n_chunks, self.start)
        timer.start()

    def extract_features(self):
        if self.args.features_name.lower() == 'mfccs':
            return self.mfccs(self.signal)
        return self.logfbes(self.signal)


def main():
    args = get_args()
    streaming_audio = StreamingAudio(args)
    streaming_audio.start()


if __name__ == '__main__':
    main()
