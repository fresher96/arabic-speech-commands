import os
import numpy as np
import pyaudio
from threading import Timer

from comet_ml import Experiment
from src.configs import get_args
from src.data import get_dataloader
from src.trainer import ModelTrainer
from src.model import *
from src.ClassDict import ClassDict


def load_model(args):
    model_constructor = globals()[args.model];
    model = model_constructor(args);
    return model;

class StreamingAudio:

    def __init__(self, args):
        self.args = args

        dataloader = get_dataloader(args)
        self.model = load_model(args)
        trainer = ModelTrainer(self.model, dataloader, args)
        trainer.load_weights()
        self.model = trainer

        self.signal_samples = args.signal_samples

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

    def start(self):
        # Read a chunk from the audio stream
        chunk_data = self.stream.read(self.chunk_size)
        # Convert the chunk of data to int16
        chunk_signal = np.fromstring(chunk_data, dtype=np.int16)
        # Discard the oldest chunk and append the new one
        self.signal = np.concatenate((self.signal[self.chunk_size:], chunk_signal))

        x = self.signal
        # Predict the received speech command
        prediction = self.model.predict(x)[0]
        # print(prediction.max())

        res_idx = self.silence_idx if prediction.max() < self.args.p_threshold else prediction.argmax()
        self.predictions = np.concatenate((self.predictions[1:], np.array(res_idx).reshape(1)))
        (values, counts) = np.unique(self.predictions, return_counts=True)
        command_idx = int(values[np.argmax(counts)])

        predicted_command = 'silence' if command_idx == self.silence_idx else ClassDict.getName(command_idx)

        print(predicted_command)

        timer = Timer(self.args.signal_len / self.args.n_chunks, self.start)
        timer.start()


def main():
    args = get_args()
    streaming_audio = StreamingAudio(args)
    print('started stream...')
    streaming_audio.start()


if __name__ == '__main__':
    main()
