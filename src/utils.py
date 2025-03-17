from scipy import signal
from scipy.io import wavfile
import os
import numpy as np
import pandas as pd
    
    
def read_wav_file(file_path):
    try:
        sample_rate, samples = wavfile.read(file_path)
        print("reading wav file ...")
        samples = samples.astype(np.float32)
        samples /= np.iinfo(np.int16).max
        response = {"status": 200, "message": "wav file read successfully", "file_name" : file_path, "sample_rate": sample_rate, "samples": samples}
        print("succesfully read wav file ...")
        return response
    except Exception as e :
        print(f"error whiule reading {file_path} file .... X")
        response = {"status": 400, "message": f"Error reading {file_path} wav file", "file_name" : file_path}
        return response
    
    
def get_duration_file(data):
    try :
        time =  (len(data["samples"]))/(data["sample_rate"])
        return time
    except Exception as e :
        print("Error while calculating time duration of file")
        return -1
    
    
    
def assign_labels_to_signal(samples, sample_rate, time_labels):
    labels = np.empty(len(samples), dtype=object)

    for start_time, end_time, label in time_labels:
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        labels[start_sample:end_sample] = label

    return labels



def split_audio_time_domain_into_windows(samples, sample_rate, time_labels, window_size_ms=100):
    
    window_size_samples = int((window_size_ms / 1000) * sample_rate)
    labels = assign_labels_to_signal(samples, sample_rate, time_labels)
    windows = []

    for start_sample in range(0, len(samples), window_size_samples):
        end_sample = start_sample + window_size_samples
        
        if end_sample > len(samples):
            end_sample = len(samples)
        
        window_labels = labels[start_sample:end_sample]
        window_samples = samples[start_sample:end_sample]
        unique_labels, counts = np.unique(window_labels, return_counts=True)
        
        if len(unique_labels) > 0:
            most_frequent_label = unique_labels[np.argmax(counts)]
        else:
            most_frequent_label = "silence"  
        
        start_time = start_sample / sample_rate
        end_time = end_sample / sample_rate
        windows.append({"start" : start_time, "end" : end_time, "label" : most_frequent_label, "samples" : window_samples})
        #windows.append((start_time, end_time, most_frequent_label))
    
    return windows

def get_pd_from_dict(dict_windows):
    try : 
        df = pd.DataFrame(dict_windows)
        return df
    except Exception as e:
        print("Error while trying to convert dict into dataframe")