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
import librosa

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


def mel_spectrogram_and_waveform_generation(checkpoint_path, text, hparams):

    # Griffin Lim iterations
    n_iter = 60
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

    taco_stft = TacotronSTFT(
        hparams.filter_length, hparams.hop_length, hparams.win_length,
        sampling_rate=hparams.sampling_rate)
    mel_decompress = taco_stft.spectral_de_normalize(mel_outputs_postnet)
    mel_decompress = mel_decompress.transpose(1, 2).data.cpu()
    spec_from_mel_scaling = 1000
    spec_from_mel = torch.mm(mel_decompress[0], taco_stft.mel_basis)
    spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
    spec_from_mel = spec_from_mel * spec_from_mel_scaling

    waveform = griffin_lim(torch.autograd.Variable(spec_from_mel[:, :, :-1]), taco_stft.stft_fn, n_iter)
    return waveform


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
                waveform = mel_spectrogram_and_waveform_generation(tacotron_checkpoint, text, hparams)
                # Create output directory
                output_name = 'output_audio_wav_griffin_lim_taco_' + checkpoint_taco_name

                os.makedirs(os.path.join(dst_dir, output_name), exist_ok=True)
                dst_wav_path = join(os.path.join(dst_dir, output_name), "{}{}.wav".format(i, file_name_suffix))
                # save
                librosa.output.write_wav(dst_wav_path, waveform[0].data.cpu().numpy(), sr=hparams.sampling_rate)
                print("Waveform {} OK".format(i))
    except FileNotFoundError:
        print("File not found.")


    print("Finished! Check out {} for generated audio samples.".format(dst_dir))
    sys.exit(0)
