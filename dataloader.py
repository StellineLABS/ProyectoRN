import os
import json
import torchaudio
import numpy as np
import pandas as pd
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import random
import csv


def make_index_dict(label_csv):
    '''
    Crea un diccionario con los indices 
    :param: label_csv: archivo con los índices de las etiquetas
    :returns: 
    '''
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
            line_count += 1
    return index_lookup

def preemphasis(signal,coeff=0.97):
    """
    Pone énfasis previo en la señal de entrada.

    :param signal: La señal a filtrar
    :param coeff: El coeficiente de preenfasis
    :returns: La señal filtrada
    """
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])

class AudiosetDataset(Dataset):
    def __init__(self, dataset_json_file, label_csv):
        """
        Dataset de audios 
        :param dataset_json_file: nombre del archivo que contiene los datos 
        :param label_csv: archivo con mapeo de nombre de todas las clases
        """
        #carga los archivos de json
        self.datapath = dataset_json_file
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)
        self.data = data_json['data']
        self.audio_conf  = {'num_mel_bins': 128, 'target_length': 1024, 'freqm': 24, 'timem': 192, 'mixup': 0.5}
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.index_dict = make_index_dict(label_csv)
        self.label_num = len(self.index_dict)
        print('Total de clases {:d}'.format(self.label_num))
        
    def _wav2fbank(self, filename):
        # mixup
        waveform, sr = torchaudio.load(filename)
        waveform = waveform - waveform.mean()
    
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)
        target_length = self.audio_conf.get('target_length')
        n_frames = fbank.shape[0]
        p = target_length - n_frames
        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]
        return fbank

    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """        
        datum = self.data[index]
        label_indices = np.zeros(self.label_num)
        fbank = self._wav2fbank(datum['wav'])
        for label_str in datum['labels'].split(','):
            label_indices[int(self.index_dict[label_str])] = 1.0
        label_indices = torch.FloatTensor(label_indices)
        fbank = torch.transpose(fbank, 0, 1)
        # this is just to satisfy new torchaudio version, which only accept [1, freq, time]
        fbank = fbank.unsqueeze(0)
        # squeeze it back, it is just a trick to satisfy new torchaudio version
        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)

        # the output fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        return fbank, label_indices
    def __len__(self):
        return len(self.data)
    