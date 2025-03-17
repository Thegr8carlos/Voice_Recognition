import matplotlib.pyplot as plt
import numpy as np
import librosa
import seaborn as sns

color_map = {
    "voice": "blue",
    "unvoice": "red",
    "silence": "green"
}

def plot_stats_df(df, feature):
    sns.boxplot(x="label", y=feature, data=df)
    plt.show()

def plot_signal_with_labels(samples, labels, sample_rate):
    time_seconds = np.arange(0, len(samples)) / sample_rate

    plt.figure(figsize=(10, 6))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        if label in color_map:
            indices = np.where(labels == label)[0]
            if len(indices) > 0:
                segment_time = time_seconds[indices]
                segment_amplitude = samples[indices]
                plt.scatter(segment_time, segment_amplitude, color=color_map[label], label=label, s=10)

    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title("Signal with Labels")
    plt.legend()
    plt.show()


def plot_signal(request):
    title = f"Signal of file {request['file_name']}"
    request["sample_rate"]
    request["samples"]
    time_seconds = np.arange(0, len(request["samples"])) / request["sample_rate"]
    plt.figure(figsize=(10, 6))
    plt.plot(time_seconds, request["samples"], label="Signal")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.legend()
    plt.show()


def plot_signal_scatter(request):
    title = f"Signal of file {request['file_name']}"
    sample_rate = request["sample_rate"]
    samples = request["samples"]
    time_seconds = np.arange(0, len(samples)) / sample_rate
    plt.figure(figsize=(10, 6))
    plt.scatter(time_seconds, samples, label="Signal", color='blue', marker='o', s=10)  # Scatter plot
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.legend()
    plt.show()
    
def plot_spectogram(D_db, sample_rate, hop_length):
    plt.figure(figsize=(8, 9))
    librosa.display.specshow(D_db, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Espectrograma ')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Frecuencia (Hz)')
    plt.show()