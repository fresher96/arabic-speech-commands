import os
import numpy as np
import pyaudio
from threading import Timer
# from keras.models import load_model
from tensorflow.keras.models import load_model

from src.configs import get_args
from src.ClassDict import ClassDict
from src.transforms import extract_features


class StreamingAudio:

    def __init__(self, args):
        self.args = args

        model_name = self.args.model + '_' + self.args.features_name
        model_path = os.path.join('..', 'output', 'models', model_name + '.model')

        self.model = load_model(model_path)

        self.signal_samples = self.args.signal_len * self.args.signal_sr

        # Create an array of length that equals to signal_samples, and initialize it with zeros
        self.signal = np.zeros(self.signal_samples)
        self.num_chunks = int(self.args.signal_len / (2 * self.args.winlen))

        silence_index = ClassDict.len()
        # Create an array that contains the last num_chunks predictions (updated continuously)
        self.predictions = np.full(self.num_chunks, silence_index)

        self.chunk_size = int(2 * self.args.winlen * self.signal_samples)

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

    def start(self):
        # Read a chunk from the audio stream
        chunk_data = self.stream.read(self.chunk_size)
        # Convert the chunk of data to int16
        chunk_signal = np.fromstring(chunk_data, dtype=np.int16)
        # Discard the oldest chunk and append the new one
        self.signal = np.concatenate((self.signal[self.chunk_size:], chunk_signal))

        # Extract features from the current signal
        x = extract_features(self.args.features_name, self.args.nfilt, self.signal)

        # Reshape the sample according to the chosen model
        if self.args.model.lower() == 'cnn':
            x = x.reshape((1, x.shape[0], x.shape[1], 1))

        # Predict the received speech command
        prediction = self.model.predict(x)[0]

        self.predictions = np.concatenate((self.predictions[1:], np.array(np.argmax(prediction)).reshape(1)))
        (values, counts) = np.unique(self.predictions, return_counts=True)
        command_idx = int(values[np.argmax(counts)])

        predicted_command = 'silence' if command_idx == ClassDict.len() else ClassDict.getName(command_idx)

        print(predicted_command)

        timer = Timer(2 * self.args.winlen, self.start)
        timer.start()


def main():
    args = get_args()
    streaming_audio = StreamingAudio(args)
    streaming_audio.start()


if __name__ == '__main__':
    main()
