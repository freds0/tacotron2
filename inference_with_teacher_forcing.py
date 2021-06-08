import os
import time
import argparse
import math
from numpy import finfo
import numpy as np

import torch
from distributed import apply_gradient_allreduce
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from model import Tacotron2
from data_utils import TextMelLoader, TextMelCollate
from loss_function import Tacotron2Loss
from logger import Tacotron2Logger
from hparams import create_hparams

from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
from plotting_utils import save_figure_to_numpy

from layers import TacotronSTFT
from audio_processing import griffin_lim
import librosa

def plot_spectrogram(spectrogram, out_path=''):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    fig.savefig(out_path)

    # data = save_figure_to_numpy(fig)
    plt.close()
    # return data


def load_model(hparams):
    model = Tacotron2(hparams).cuda()
    if hparams.fp16_run:
        model.decoder.attention_layer.score_mask_value = finfo('float16').min

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    return model


def warm_start_model(checkpoint_path, model, ignore_layers):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = checkpoint_dict['state_dict']
    if len(ignore_layers) > 0:
        model_dict = {k: v for k, v in model_dict.items()
                      if k not in ignore_layers}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict)
    return model


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration

def spec_to_waveform(taco_stft, mel_outputs_postnet, n_iter=60):
    mel_decompress = taco_stft.spectral_de_normalize(mel_outputs_postnet).unsqueeze(0)
    mel_decompress = mel_decompress.transpose(1, 2).data.cpu()
    spec_from_mel_scaling = 1000
    spec_from_mel = torch.mm(mel_decompress[0], taco_stft.mel_basis)
    spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
    spec_from_mel = spec_from_mel * spec_from_mel_scaling
    waveform = griffin_lim(torch.autograd.Variable(spec_from_mel[:, :, :-1]), taco_stft.stft_fn, n_iter)
    return waveform[0]

def infer(output_directory, checkpoint_path, warm_start, hparams, debug=False):
    """Inference with teaching force

    Params
    ------
    output_directory (string): directory to the spectrograms
    checkpoint_path(string): checkpoint path
    hparams (object): comma separated list of "name=value" pairs.
    """

    os.makedirs(output_directory, exist_ok=True) 
    taco_stft = TacotronSTFT(
                        hparams.filter_length, hparams.hop_length, hparams.win_length,
                        sampling_rate=hparams.sampling_rate)

    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    model = load_model(hparams)
    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=hparams.weight_decay)

    if hparams.fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level='O2')

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    return_file_name = True

    trainset = TextMelLoader(hparams.training_files, hparams, return_file_name=return_file_name)
    collate_fn = TextMelCollate(hparams.n_frames_per_step,  return_file_name=return_file_name)


    train_sampler = None

    train_loader = DataLoader(trainset, num_workers=1, shuffle=False,
                              sampler=train_sampler,
                              batch_size=hparams.batch_size, 
                              pin_memory=False, 
                              collate_fn=collate_fn)

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    if checkpoint_path is not None:
        if warm_start:
            model = warm_start_model(
                checkpoint_path, model, hparams.ignore_layers)
        else:
            model, optimizer, _learning_rate, iteration = load_checkpoint(
                checkpoint_path, model, optimizer)
            if hparams.use_saved_learning_rate:
                learning_rate = _learning_rate
            iteration += 1  # next iteration is iteration + 1
            epoch_offset = max(0, int(iteration / len(train_loader)))

    model.eval()
    print('Starting ...')
    for i, batch in enumerate(tqdm(train_loader)):
            x, y = model.parse_batch(batch[:][:-1])
            files_name = batch[:][-1]
            mel_outputs, mel_outputs_postnet, _, alignments = model(x)

            _, _, mel_expected_padded, _, mel_lengths = x

            for idx in range(mel_outputs_postnet.size(0)):

                name = os.path.basename(files_name[idx]).replace(".wav", '')
                mel_padded = mel_outputs_postnet[idx]
                mel_length = mel_lengths[idx]
                mel = mel_padded[:, :mel_length]
                np.save(os.path.join(output_directory, name+'.npy'), mel.detach().cpu().numpy())

                if debug:
                    print("Debug Mode ON: Saving Wave files and Spectrograms Plot in:", output_directory)
                    # plot audios
                    librosa.output.write_wav(os.path.join(output_directory, name+'.wav'), spec_to_waveform(taco_stft, mel).detach().cpu().numpy(), sr=hparams.sampling_rate)
                    librosa.output.write_wav(os.path.join(output_directory, name+'_padded.wav'), spec_to_waveform(taco_stft, mel_padded).detach().cpu().numpy(), sr=hparams.sampling_rate)
                    librosa.output.write_wav(os.path.join(output_directory, name+'_expected_padded.wav'), spec_to_waveform(taco_stft, mel_expected_padded[idx]).detach().cpu().numpy(), sr=hparams.sampling_rate)
                    # plot figures
                    plot_spectrogram(mel.detach().cpu().numpy(), )
                    plot_spectrogram(mel_padded.detach().cpu().numpy(), os.path.join(output_directory, name+'_padded.png'))
                    plot_spectrogram(mel_expected_padded[idx].detach().cpu().numpy(), os.path.join(output_directory, name+'_expect_padded.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save mel spectrograms',  default='mels_specs-test/')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--debug', type=bool, default=False,
                        required=False, help='Active Degub Mode')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')

    args = parser.parse_args()
    hparams = create_hparams(args.hparams)

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    print("FP16 Run:", hparams.fp16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Distributed Run:", hparams.distributed_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)

    infer(args.output_directory, args.checkpoint_path,
          args.warm_start, hparams, args.debug)
