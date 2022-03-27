#!/usr/bin/env python
import os, pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow_io as tfio
import keras

from keras import layers, models, preprocessing
from IPython import display

# Set seed for experiment reproducibility
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

URL = "https://drive.upm.es/s/tDxI39MyWb0woIw/download"
data_dir = pathlib.Path('data/Samples')

if not data_dir.exists():
    keras.utils.get_file(
        'Samples.zip',
        URL,
        extract = True,
        cache_dir = '.', cache_subdir = 'data'
    )

# Functions
SAMPLE_FREQUENCE = 16000
LABELS = 64
SAMPLES = 128
WINDOW_ANALISYS_SIZE = 200 # in miliseconds
AUTOTUNE = tf.data.AUTOTUNE

def open_audio_file(file=None):
    audio_bin = tf.io.read_file(file)
    audio = tfio.audio.decode_flac(audio_bin, shape=None, dtype= tf.int16, name=None)
    audio_f = tf.cast(tf.squeeze(audio), dtype = tf.float32)
    audio_f = audio_f[:SAMPLES]

    return audio_f

def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-3]

def get_waveform_and_label(file_path):
    label = get_label(file_path)
    waveform = open_audio_file(file_path)
    
    return waveform, label

def get_spectrogram(waveform):
  zero_padding = tf.zeros([SAMPLES] - tf.shape(waveform), dtype=tf.float32)

  # Concatenate audio with padding so that all audio clips will be of the 
  # same length
  waveform = tf.cast(waveform, tf.float32)
  equal_length = tf.concat([waveform, zero_padding], 0)
  spectrogram = tf.signal.stft(
      equal_length, frame_length=255, frame_step=128)
  spectrogram = tf.abs(spectrogram)

  return spectrogram

def plot_spectrogram(spectrogram, ax):
  # Convert to frequencies to log scale and transpose so that the time is
  # represented in the x-axis (columns).
  log_spec = np.log(spectrogram.T)
  height = log_spec.shape[0]
  width = log_spec.shape[1]
  X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec)

def get_spectrogram_and_label_id(audio, label):
  spectrogram = get_spectrogram(audio)
  spectrogram = tf.expand_dims(spectrogram, -1)
  label_id = tf.argmax(label==tipo_audio)
  return spectrogram, label_id

def preprocess_dataset(files):
  files_ds = tf.data.Dataset.from_tensor_slices(files)
  output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
  output_ds = output_ds.map(
      get_spectrogram_and_label_id,  num_parallel_calls=AUTOTUNE)
  return output_ds
