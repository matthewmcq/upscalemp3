import librosa
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import gc
import soundfile as sf
import tqdm
import main
import normalize_
import preprocessing
import noisereduce as nr


def combine_spectrograms(magnitudes, phases, sample_rate=44100, nfft=preprocessing.N_FFT, hop_length=preprocessing.HOP_LENGTH):

    # Calculate overlap in frames
    overlap_samples = 0.5 * sample_rate  # 0.5 seconds
    overlap_frames = int(np.ceil(overlap_samples / hop_length))

    # Create window for blending
    window = np.hanning(2 * overlap_frames)[overlap_frames:]  # Second half of the Hann window

    # Initial spectrogram
    #print(magnitudes.shape, phases.shape)
    #print(magnitudes[0].shape, phases[0].shape)
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

    # After the loop, check sizes
    #print(combined_magnitudes.shape, combined_phases.shape)  # They must match

    return combined_magnitudes, combined_phases


def modified_griffin_lim(magnitude_spectrogram, initial_phase, n_iter=200, phase_viz=False):
    
    phase = initial_phase
    # Iterative process
    for i in tqdm.tqdm(range(n_iter), desc="Modified Griffin-Lim Iterations"):
        # Inverse STFT
        reconstructed_signal = librosa.istft(magnitude_spectrogram * np.exp(1j * phase), 
                                             hop_length=preprocessing.HOP_LENGTH, n_fft=preprocessing.N_FFT)

        # STFT to get a new spectrogram with the original magnitude
        _, phase = librosa.magphase(librosa.stft(reconstructed_signal, n_fft=preprocessing.N_FFT, hop_length=preprocessing.HOP_LENGTH))

    print("Iterations complete")
    gl = librosa.istft(magnitude_spectrogram * np.exp(1j * phase), hop_length=preprocessing.HOP_LENGTH, n_fft=preprocessing.N_FFT)

    if phase_viz:
        plot_phase(phase, 44100, preprocessing.HOP_LENGTH)
        plot_phase(initial_phase, 44100, preprocessing.HOP_LENGTH)

    # Final reconstruction
    normal = librosa.istft(magnitude_spectrogram * np.exp(1j * initial_phase), hop_length=preprocessing.HOP_LENGTH, n_fft=preprocessing.N_FFT)
    return gl, normal

def plot_phase(data, sr, hop_length):
    #print(data.shape)
    plt.figure(figsize=(12, 8))

    # 1. Direct Phase Visualization
    plt.subplot(2, 1, 1)
    plt.imshow(data, aspect='auto', origin='lower', cmap='hsv')
    plt.colorbar()
    plt.title("Direct Phase Visualization")
    plt.xlabel("Time Frame")
    plt.ylabel("Frequency Bin")

    # 2. Unwrapped Phase Visualization
    unwrapped_phase = np.unwrap(data, axis=0)  # Unwrap along frequency axis
    plt.subplot(2, 1, 2)
    plt.imshow(unwrapped_phase, aspect='auto', origin='lower', cmap='twilight_shifted')
    plt.colorbar()
    plt.title("Unwrapped Phase Visualization")
    plt.xlabel("Time Frame")
    plt.ylabel("Frequency Bin")

    plt.tight_layout()
    plt.show()

def compare_modifiedgl_to_single_istft(gl, normal):

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

def postprocess(mags, phases, visualize=True):
    # 6. Stitch the spectrograms together along the time axis
    combined_magnitudes, combined_phases = combine_spectrograms(mags, phases)

    # 7. Run ISTFT with modified griffin lim algorithm on the stitched spectrogram to get the output audio
    #    |-> Modify griffin lim algorithm to use/start with the phase from the output of the model (should try with and without)
    # 8. Write output audio to file
    output_audio_gl, output_audio_normal = modified_griffin_lim(combined_magnitudes, combined_phases)

    reduce_gl = nr.reduce_noise(y=output_audio_gl, sr=44100, n_fft=preprocessing.N_FFT, hop_length=preprocessing.HOP_LENGTH)
    reduce_normal = nr.reduce_noise(y=output_audio_normal, sr=44100, n_fft=preprocessing.N_FFT, hop_length=preprocessing.HOP_LENGTH)

    if visualize:
        compare_modifiedgl_to_single_istft(output_audio_gl, output_audio_normal)
        compare_modifiedgl_to_single_istft(reduce_gl, reduce_normal)

        

    return output_audio_gl, output_audio_normal, reduce_gl, reduce_normal