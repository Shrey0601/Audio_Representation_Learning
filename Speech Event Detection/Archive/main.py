import argparse
import subprocess
import sys
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import Precision, Recall
import tensorflow.keras.backend as K
from tensorflow import keras
import librosa
import python_speech_features
import numpy as np
import textgrids
from sklearn import preprocessing
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import load

input_path = ""
output_path = ""

def parse():
    # Create a parser object
    parser = argparse.ArgumentParser(description="A simple command-line argument example")

    # Add the -I and -o options
    parser.add_argument("-i", "--input", help="Input file or directory")
    parser.add_argument("-o", "--output", help="Output file or directory")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the values of the -I and -o options
    input_path = args.input
    output_path = args.output

    # Use the values in your code
    return input_path, output_path

if __name__ == "__main__":
    input_path, output_path = parse()
    
model = load_model("model2.h5")
scaler = load("scaler2.joblib")

preemphasis_coef = 0.97
num_nfft = 512
frame_step = 0.01
frame_length = 0.025
n_frames = 32
num_features = 32

audio_file = input_path

dataset_valid = []

# Load samples:
input_signal, fs = librosa.load(audio_file)


stride = int(15)

# Extract logfbank features:
features_logfbank_valid = python_speech_features.base.logfbank(signal=input_signal, samplerate=fs, winlen=frame_length, winstep=frame_step, nfilt=num_features, 
                                                            nfft=num_nfft, lowfreq=0, highfreq=None, preemph=preemphasis_coef)


spectrogram_image_valid = np.zeros((n_frames, n_frames))
for j in range(int(np.floor(features_logfbank_valid.shape[0] / n_frames))):
    spectrogram_image_valid = features_logfbank_valid[j * n_frames:(j + 1) * n_frames]
    dataset_valid.append((0, spectrogram_image_valid))

if stride - len(dataset_valid) > 0:
    for i in range(stride - len(dataset_valid)):
        dataset_valid.append((0, np.zeros((n_frames, n_frames))))
        
print(dataset_valid)

# Split dataset on train and test:
X_valid = list()
for i in range(len(dataset_valid)):
    X_valid.append(dataset_valid[i][1])
    
X_valid = np.array(X_valid)

# Reshaping for scaling:
X_valid = X_valid.reshape(X_valid.shape[0], X_valid.shape[1] * X_valid.shape[2])

# Scale data:
X_valid = scaler.transform(X_valid)

print(X_valid)

# And reshape back:
X_valid = X_valid.reshape(X_valid.shape[0], n_frames, n_frames)

print(X_valid)

# Reshape data for convolution layer:
X_valid_reshaped = X_valid[:int(np.floor(X_valid.shape[0] / stride) * stride)]


X_valid_reshaped = X_valid_reshaped.reshape((int(X_valid_reshaped.shape[0] / stride), stride, n_frames, n_frames, 1))

print(X_valid_reshaped)

prediction = model.predict(X_valid_reshaped)

print(prediction)

predicted_label = np.zeros(prediction.shape[1])
predicted_proba = np.zeros(prediction.shape[1])
ind = 0
for i in range(prediction.shape[1]):
    if prediction[0][i][0] >= prediction[0][i][1]:
        predicted_label[ind] = 0
        predicted_proba[ind] = prediction[0][i][0]
    else:
        predicted_label[ind] = 1
        predicted_proba[ind] = prediction[0][i][1]
    ind = ind + 1
        
print(predicted_label)

predicted_label_widely = np.zeros(predicted_label.shape[0] * n_frames)
ind_start = 0
ind_stop = n_frames
shift_step = n_frames
for i in range(predicted_label.shape[0]):
    predicted_label_widely[ind_start:ind_stop] = predicted_label[i]
    ind_start = ind_start + shift_step
    ind_stop = ind_stop + shift_step

label_timeseries = np.zeros(input_signal.shape[0])
begin = int(0)
end = int(frame_length * fs)
shift_step = int(frame_step * fs)
for i in range(predicted_label_widely.shape[0]):
    label_timeseries[begin:end] = predicted_label_widely[i]
    begin = begin + shift_step
    end = end + shift_step
print(label_timeseries)

Ns = input_signal.shape[0]
Ts = 1 / fs  
t = np.arange(Ns) * Ts

prev_label = 0
curr_label = 0
test = 0
start = 0
end = 0
output = []

for i in range(len(label_timeseries)):
    if label_timeseries[i] == 1:
        curr_label = 1
        if prev_label == 0:
            start = t[i]
    else:
        curr_label = 0
        if prev_label == 1:
            end = t[i]
            output.append([start, end])
            test = test + 1
    prev_label = curr_label
        
if curr_label == 1:
    end = t[i]
    output.append([start, end])
    test = test + 1
            
print(output)

import csv

fields = ['start', 'end']
with open(output_path, 'w') as f:
    write = csv.writer(f)
    write.writerow(fields)
    write.writerows(output)