import numpy as np

def get_signal_energy(signal):
    return np.sum(np.square(signal))

def get_signal_mean(signal):
    return np.mean(signal)

def get_signal_potencia(signal):
    return get_signal_energy(signal) / len(signal)

def get_signal_zero_crossing_rate(signal):
    zero_crossings = np.diff(np.sign(signal)) != 0
    return np.sum(zero_crossings) / len(signal)
