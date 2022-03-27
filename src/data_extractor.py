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
PATH = 'data/Samples'
data_dir = pathlib.Path(PATH)

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
    audio = tfio.audio.decode_wav(audio_bin, shape=None, dtype= tf.int16, name=None)
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
  label_id = tf.argmax(label==audio_labels)
  return spectrogram, label_id

def preprocess_dataset(files):
  files_ds = tf.data.Dataset.from_tensor_slices(files)
  output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
  output_ds = output_ds.map(
      get_spectrogram_and_label_id,  num_parallel_calls=AUTOTUNE)
  return output_ds

audio_labels = {
	"pathology": 'ALS',
	"control": 'HC'
}
train_data = PATH + '/HC/*'
test_data = PATH + '/ALS/*'

train_files = tf.io.gfile.glob(train_data)
filenames_test = tf.io.gfile.glob(test_data)

exp = len(train_files)
num = round(exp/10)
n_train = exp - num
filenames_train = train_files[:n_train]
filenames_validation = train_files[n_train:exp]

filenames_train1 = tf.random.shuffle(filenames_train)
filenames_test  = tf.random.shuffle(filenames_test)
filenames_val   = tf.random.shuffle(filenames_validation)

filenames_train = filenames_train1[:40]
filenames_train2 = filenames_train1[40:]
filenames_val = filenames_val[:4]
filenames_test = filenames_test[:8]


num_samples_train = len(filenames_train)
num_samples_train2 = len(filenames_train2)
num_samples_test  = len(filenames_test)
num_samples_val  = len(filenames_val)

print('Numero total de casos',exp)
print('Numero total de casos para el entrenamiento del primer modelo:', num_samples_train)
print('Numero total de casos para el entrenamiento del segundo modelo:', num_samples_train2 )
print('Numero total de casos para test:', num_samples_test)
print('Numero total de casos para validacion:', num_samples_val)
print('Etiquetas: ', audio_labels)

files_ds  = tf.data.Dataset.from_tensor_slices(filenames_train)
waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)

rows = 3
cols = 3
n = rows*cols
fig, axes = plt.subplots(rows, cols, figsize=(18, 20))
for i, (audio, label) in enumerate(waveform_ds.take(n)):
  r = i // cols
  c = i % cols
  ax = axes[r][c]
  ax.plot(audio.numpy())
  ax.set_yticks(np.arange(-40, 45, 10))
  label = label.numpy().decode('utf-8')
  ax.set_title(label)

plt.show()
