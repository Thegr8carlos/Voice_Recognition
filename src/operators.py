import librosa
import numpy as np

def calculate_spectrogram(signal, sample_rate, n_fft, hop_length):
  D = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
  magnitude = np.abs(D)
  D_db = librosa.amplitude_to_db(magnitude, ref=np.max)
  return D_db



