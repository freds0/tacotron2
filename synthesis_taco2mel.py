# coding: utf-8
"""
Synthesis waveform from trained WaveNet.

usage: synthesis_taco2griffin_lim.py [options] <dst_dir>

options:
    --hparams=<parmas>                Hyper parameters [default: ].
    -i, --input-file=<p>                  Input txt file path.
    -f, --file-name-suffix=<s>            File name suffix [default: ].
    -t, --tacotron-checkpoint=<p>         Tacotron Checkpoint Path
    -h, --help               Show help message.
"""
from docopt import docopt

import sys
import os
from os.path import dirname, join, basename, splitext
import torch
import numpy as np

#################################################################
# Tacotron Methods
################################################################

# from Tacotron Modules
from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
import audio_processing


def mel_spectrogram_generation(checkpoint_path, text, hparams):

    # #### Load model from checkpoint
    model = load_model(hparams)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    _ = model.eval()


    # #### Prepare text input
    #text = "amor é fogo que arde sem se ver."
    sequence = np.array(text_to_sequence(text, ['basic_cleaners']))[None, :]
    sequence = torch.autograd.Variable(
        torch.from_numpy(sequence)).cuda().long()


    # #### Decode text input

    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
    return mel_outputs_postnet.data.cpu()


#################################################################
# Main
################################################################

if __name__ == "__main__":
    args = docopt(__doc__)
    print("Command line args:\n", args)
    tacotron_checkpoint  = args["--tacotron-checkpoint"]
    dst_dir = args["<dst_dir>"]

    input_file_path = args["--input-file"]
    file_name_suffix = args["--file-name-suffix"]

    # Create output directory
    os.makedirs(dst_dir, exist_ok=True)

    checkpoint_taco_name = splitext(basename(tacotron_checkpoint))[0].replace('checkpoint_', '')

    hparams = create_hparams("distributed_run=False,mask_padding=False")
    hparams.sampling_rate = 22050
    hparams.filter_length = 1024
    hparams.hop_length = 256
    hparams.win_length = 1024
    # Mel-spectrogram generation with tacotron
    mel_spectrograms_list = []
    try:
        with open(input_file_path) as f:
            content =  f.read().splitlines()
            for i, text in enumerate(content):
                print("Generating waveform " + str(i))
                mel = mel_spectrogram_generation(tacotron_checkpoint, text, hparams)
                # Create output directory
                output_name = 'output_audio_wav_griffin_lim_taco_' + checkpoint_taco_name

                os.makedirs(os.path.join(dst_dir, output_name), exist_ok=True)
                dst_mel_path = join(os.path.join(dst_dir, output_name), "{}{}.npy".format(i, file_name_suffix))
                # save
                np.save(dst_mel_path, mel)
                print("Mel {} OK".format(i))
    except FileNotFoundError:
        print("File not found.")


    print("Finished! Check out {} for generated audio samples.".format(dst_dir))
    sys.exit(0)
