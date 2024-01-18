import librosa
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import gc
import soundfile as sf
import tqdm
import main
import normalize
import preprocessing

def combine_spectrograms(prediction, sample_rate=44100, nfft=2048, hop_length=512):
    magnitudes = prediction[..., 0]
    phases = prediction[..., 1]

    # Calculate overlap in frames
    overlap_samples = 0.5 * sample_rate  # 0.5 seconds
    overlap_frames = int(np.ceil(overlap_samples / hop_length))

    # Create window for blending
    window = np.hanning(2 * overlap_frames)[overlap_frames:]  # Second half of the Hann window

    # Initial spectrogram
    combined_magnitudes = magnitudes[0]
    combined_phases = phases[0]

    for i in range(1, len(magnitudes)):
        # Apply window to the overlapping regions
        combined_magnitudes[:, -overlap_frames:] *= window
        combined_magnitudes[:, -overlap_frames:] += magnitudes[i][:, :overlap_frames] * (1 - window)
        combined_phases[:, -overlap_frames:] *= window
        combined_phases[:, -overlap_frames:] += phases[i][:, :overlap_frames] * (1 - window)

        # Combine with non-overlapping parts
        combined_magnitudes = np.concatenate((combined_magnitudes, magnitudes[i][:, overlap_frames:]), axis=1)
        combined_phases = np.concatenate((combined_phases, phases[i][:, overlap_frames:]), axis=1)

    return combined_magnitudes, combined_phases


def modified_griffin_lim(magnitude_spectrogram, initial_phase, n_iter=100):
    # Initialize the phase
    phase = initial_phase

    # Iterative process
    for i in range(n_iter):
        # Inverse STFT
        reconstructed_signal = librosa.istft(magnitude_spectrogram * np.exp(1j * phase))

        # STFT to get a new spectrogram with the original magnitude
        _, phase = librosa.magphase(librosa.stft(reconstructed_signal, n_fft=preprocessing.N_FFT, hop_length=preprocessing.HOP_LENGTH))

    # Final reconstruction
    normal = librosa.istft(magnitude_spectrogram * np.exp(1j * initial_phase))
    return librosa.istft(magnitude_spectrogram * np.exp(1j * phase)), normal

def compare_modifiedgl_to_single_istft(magnitude_spectrogram, initial_phase, n_iter=100):
    # Modified griffin lim version
    gl, normal = modified_griffin_lim(magnitude_spectrogram, initial_phase, n_iter)
    

    # visualize the two waveforms and their difference (should be zero)
    plt.figure(figsize=(21, 5))
    plt.subplot(1, 3, 1)
    plt.title("Modified Griffin-Lim")
    plt.plot(gl)
    plt.subplot(1, 3, 2)
    plt.title("Normal ISTFT")
    plt.plot(normal)
    plt.show()
    plt.subplot(1, 3, 3)
    plt.title("Difference")
    plt.plot(gl-normal)
    plt.show()

def postprocess(prediction, visualize=True):
    # 6. Stitch the spectrograms together along the time axis
    combined_magnitudes, combined_phases = combine_spectrograms(prediction)

    # 7. Run ISTFT with modified griffin lim algorithm on the stitched spectrogram to get the output audio
    #    |-> Modify griffin lim algorithm to use/start with the phase from the output of the model (should try with and without)
    # 8. Write output audio to file
    output_audio_gl, output_audio_normal = modified_griffin_lim(combined_magnitudes, combined_phases)
    if visualize:
        compare_modifiedgl_to_single_istft(combined_magnitudes, combined_phases)

    return output_audio_gl, output_audio_normal