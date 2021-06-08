import argparse
import glob
import os
from os import makedirs
from os.path import basename, join, exists, isdir, getsize, dirname
import random
import numpy as np
import torch.utils.data

import layers
from utils import load_wav_to_torch, load_filepaths_and_text

from hparams import create_hparams

class mel_generator():
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, hparams):
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        random.seed(hparams.seed)


    def get_mel(self, filename):

        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.stft.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.stft.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)
        return melspec


def execute(input_dir, output_dir, hparams):
    
    mg = mel_generator(hparams)
    if not exists(output_dir):
        makedirs(output_dir)

    for filepath in sorted(glob.glob(input_dir + '*.wav')):
        mel = mg.get_mel(filepath)
        filename = basename(filepath).split('.')[0]
        np.save(join(output_dir, filename), mel.numpy())

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='./')
    parser.add_argument('--input_dir', default='./daps/wavs22/')
    parser.add_argument('--output_dir', default='test_mels22/')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')
    args = parser.parse_args()

    hparams = create_hparams(args.hparams)

    execute(join(args.base_dir, args.input_dir), join(args.base_dir, args.output_dir), hparams)

if __name__ == "__main__":
    main()
