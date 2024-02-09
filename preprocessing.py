import librosa
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import gc
import soundfile as sf
import tqdm

if not os.path.exists('coeffs'):
  os.makedirs('coeffs')

# CONSTANTS -- MODEL ARCHITECTURE DEPENDS ON THESE => DO NOT CHANGE!
N_FFT = 2048
HOP_LENGTH = 256

# Preprocessing

#1. Split mp3 into 1 second chunks with 0.5 second overlap (if \in [1, 1.5]s, pad with zeros at beginning and end to 1.5 seconds)
    
def split_audio(input_path, output_path):
    # Load audio file in stereo
    y, sr = librosa.load(input_path, sr=44100, mono=False)
    # y has shape (2, num_samples) where y[0] is the left channel and y[1] is the right channel
    
    if y.shape[1] > 44100 * 1.5:
        for i in range(0, y.shape[1] - 44100, 44100 // 2):
            # Write both channels
            sf.write(output_path + str(i) + '.wav', y[:, i:i+44100].T, sr)
    elif y.shape[1] < 44100:
        y = np.pad(y, ((0, 0), (44100 - y.shape[1], 44100 - y.shape[1])), 'constant')
        sf.write(output_path + '0.wav', y.T, sr)
    else:
        sf.write(output_path + '0.wav', y.T, sr)

# Helper function to extract the numerical part of the filename
def extract_number(filename):
    # Remove the file extension and convert to integer
    return int(filename.split('.')[0])
# 2. For each chunk, compute the STFT with hop len 512 and nfft 2048
        
def get_spectrogram_segments(filepath_mp3, n_fft=N_FFT, hop_length=HOP_LENGTH):
    print("getting spectrogram segments")
    magnitudes = []
    phases = []

    # Get all .wav files and sort them by the numerical part of the filename
    mp3_files = [f for f in os.listdir(filepath_mp3) if f.endswith('.wav')]
    mp3_files.sort(key=extract_number)
    length = len(mp3_files)

    for filename in tqdm.tqdm(mp3_files, desc="Processing MP3 files"):
        # Load the stereo file
        audio, sr = librosa.load(os.path.join(filepath_mp3, filename), sr=44100, mono=False)
        # Process each channel independently
        for channel in range(audio.shape[0]):
            S_mp3 = librosa.stft(audio[channel], n_fft=n_fft, hop_length=hop_length)
            magnitudes.append(np.abs(S_mp3))
            phases.append(np.angle(S_mp3))
        
    print("unifying dimensions")
    # Assuming you have a function defined to unify dimensions of magnitude and phase
    magnitudes, phases = unify_dimensions(magnitudes, phases)

    return magnitudes, phases


def unif_pad_array(arr, max_time, max_freq):
        time_pad_length = max_time - arr.shape[1]
        freq_pad_length = max_freq - arr.shape[0]
        return np.pad(arr, ((0, freq_pad_length), (0, time_pad_length)))

def unify_dimensions(magnitudes, phases):
    max_time_length = max([mag.shape[1] for mag in magnitudes + phases])
    max_freq_length = max([mag.shape[0] for mag in magnitudes + phases])
    print(max_time_length, max_freq_length)

    unified_mags = []
    #for mag in tqdm(magnitudes, desc="Processing Magnitudes"):
    for mag in magnitudes:
        mag_padded = unif_pad_array(mag, max_time_length, max_freq_length)
        unified_mags.append(mag_padded)

    unified_phases = []
    #for phase in tqdm(phases, desc="Processing Phases"):
    for phase in phases:
        phase_padded = unif_pad_array(phase, max_time_length, max_freq_length)
        unified_phases.append(phase_padded)

    print(unified_mags[0].shape)
    print("Converting to numpy arrays and returning")
    return np.array(unified_mags), np.array(unified_phases)




def comb_pad_array(arr, max_time, max_freq):
        time_pad_length = max_time - arr.shape[1]
        freq_pad_length = max_freq - arr.shape[0]
        return np.pad(arr, ((0, freq_pad_length), (0, time_pad_length)))


def wrapped_phase_difference(phase1, phase2):
          diff = tf.abs(phase1 - phase2)
          return tf.minimum(diff, 1.0 - diff)

def combine_magnitude_phase( magnitudes_mp3, phases_mp3):

    print("Finding max lengths for padding")
    max_freq_length, max_time_length = max(
        (mag.shape[0], mag.shape[1]) for mag in
        (*magnitudes_mp3, *phases_mp3)
    )
    print(max_time_length, max_freq_length)

    combined_mp3 = []

    for mag, phase in zip(magnitudes_mp3, phases_mp3):
        mag_padded = comb_pad_array(mag, max_time_length, max_freq_length)
        phase_padded = comb_pad_array(phase, max_time_length, max_freq_length)
        combined_mp3.append(np.stack([mag_padded, phase_padded], axis=-1))

    print("Converting to numpy arrays")
    combined_mp3 = np.array(combined_mp3)
    return combined_mp3 

def preprocess(input_path):
    if not os.path.exists('split'):
        os.makedirs('split')
    # 1. Split mp3 into 1 second chunks with 0.5 second overlap (if \in [1, 1.5]s, pad with zeros at beginning and end to 1.5 seconds)
    print("splitting audio")
    split_audio(input_path, 'split/')
    # 2. For each chunk, compute the STFT with hop len 512 and nfft 2048
    print("getting spectrogram segments")
    magnitudes_mp3, phases_mp3 = get_spectrogram_segments('split/')
    # 3. Pad magnitude and phase to be of shape 1025, 87
    print("combining magnitudes and phases")
    combined_mp3 = combine_magnitude_phase(magnitudes_mp3, phases_mp3)
    return combined_mp3