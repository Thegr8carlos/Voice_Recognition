from operators import calculate_spectrogram
from utils import read_wav_file, assign_labels_to_signal,  split_audio_time_domain_into_windows, get_pd_from_dict
from plots import plot_signal, plot_spectogram, plot_signal_with_labels, plot_stats_df
from labels import audio1_labels
from patterns_extractions import get_signal_energy, get_signal_mean, get_signal_potencia, get_signal_zero_crossing_rate
from models import decision_tree,  naive_bayes_classifier
data = read_wav_file("audio1.wav")
#plot_signal(data)

n_fft = 1024
hops = 100
spectogram = calculate_spectrogram(data["samples"], data["sample_rate"], n_fft, hops)
plot_spectogram(spectogram, data["sample_rate"], hops)

#print(spectogram.shape)
#print(len(data["samples"]))
#labels_audio_1 = assign_labels_to_signal(data["samples"], data["sample_rate"], audio1_labels)

#plot_signal_with_labels(data["samples"], labels_audio_1, data["sample_rate"])


windows = split_audio_time_domain_into_windows(data["samples"], data["sample_rate"], audio1_labels)

df = get_pd_from_dict(windows)
print(f"Original Dataset without new features \n {df}")

df["energy"] = df["samples"].apply(get_signal_energy)
df["mean"] = df["samples"].apply(get_signal_mean)
df["potencia"] = df["samples"].apply(get_signal_potencia)
df["zero_crossing_rate"] = df["samples"].apply(get_signal_zero_crossing_rate)


print(f"After feature extraction \n {df}")

stats = df.groupby("label")[["energy", "mean", "potencia", "zero_crossing_rate"]].agg(["mean", "std", "min", "max"])

print(f"Statistics per label: \n{stats}")

#plot_stats_df(df, "potencia")
#plot_stats_df(df, "mean")


decision_tree(df)
naive_bayes_classifier(df)
# for w in windows:
#     print(f"Ventana {w["label"]} \n Comienzo :{w["start"]} Final {w["end"]} Samples {len(w["samples"])}")