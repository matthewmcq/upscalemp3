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

@tf.keras.saving.register_keras_serializable()
class DynamicResizeLayer(tf.keras.layers.Layer):
    def call(self, inputs, target_shape):
        # Resize the input to the target shape
        return tf.image.resize(inputs, size=(target_shape[1], target_shape[2]))
    def get_config(self):
        # Return an empty config dictionary since this layer has no configurable parameters
        return {}

@tf.keras.saving.register_keras_serializable()
def log_spectral_distance():
    @tf.keras.saving.register_keras_serializable()
    def loss(y_true, y_pred):
        # Extract the first channel for magnitude
        y_true_mag = y_true[..., 0:1]
        y_pred_mag = y_pred[..., 0:1]

        # Calculate the difference per item
        diff_per_item = y_true_mag - y_pred_mag

        norm_per_item = tf.norm(diff_per_item, axis=[1, 2])

        # Compute the mean of these norms
        mean_norm = tf.reduce_mean(norm_per_item)

        return mean_norm

    return loss


custom_objects = {
    'DynamicResizeLayer': DynamicResizeLayer, # Used for residual encoder blocks
    'log_spectral_distance': log_spectral_distance(), # custom loss function
}

def normalize(y):
    """Normalize a numpy array to the range [0, 1]."""
    y_min, y_max = y.min(), y.max()
    if y_max == y_min:
        # Avoid division by zero
        return np.ones_like(y)
    return (y - y_min) / (y_max - y_min)


def extrapolate_frequency_content(data, boundary=700, degree=3, decay_rate=0.999, load_data=False):
    if load_data:
      print("loading transients")
      modified_data = np.load("coeffs/transients.npy", allow_pickle=True)
      return modified_data

    magnitude = data[..., 0]
    phase = data[..., 1]

    modified_magnitude = np.copy(magnitude)
    modified_phase = np.copy(phase)

    for i in tqdm.tqdm(range(magnitude.shape[0]), total=magnitude.shape[0], desc="adding frequencies to transients"):  # Iterate over examples
        for j in range(magnitude.shape[2]):  # Iterate over frames
            frame_magnitude = magnitude[i, :, j]

            # Get known lower frequencies and their indices
            x = np.arange(boundary)
            y = frame_magnitude[:boundary]

            # Fit a polynomial to the known data
            coeffs = np.polyfit(x, y, degree)
            polynomial = np.poly1d(coeffs)

            # Extrapolate to the higher frequency bins
            x_high = np.arange(boundary, magnitude.shape[1])
            y_high = polynomial(x_high)

            # Normalize the extrapolated values to [0, 1]
            y_high_normalized = normalize(y_high)

            # Determine the starting magnitude for exponential decay
            avg_bins = 16
            start_magnitude = np.max(frame_magnitude[boundary-avg_bins:boundary])
            y_high_masked = y_high_normalized * start_magnitude
            decay_length = len(y_high_masked)
            decay_factor = np.power(decay_rate, np.arange(decay_length))

            # Apply the decay to the extrapolated values
            y_high_masked *= decay_factor


            # Update the frame with the masked extrapolated values
            condition = modified_magnitude[i, boundary:, j] != 0  # Identify non-zero values
            avg_values = (modified_magnitude[i, boundary:, j] + y_high_masked) / 2  # Compute average values

            modified_magnitude[i, boundary:, j] = np.where(condition, avg_values, y_high_masked)


    modified_phase = phase
    # Combine magnitude and modified phase (phase remains unchanged)
    modified_data = np.stack([modified_magnitude, modified_phase], axis=-1)
    print("saving extended transients")
    np.save("coeffs/transients.npy", modified_data)
    print("saved modified data")
    return modified_data


def plot_phase(data, sr, hop_length):
    print(data.shape)
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


def visualize_polyfit(combined_mp3, harmonic_mp3):
    # Convert your magnitude and phase to complex spectrogram
    mp3_spectrograms = combined_mp3[0:10][...,0] * np.exp(1j * combined_mp3[0:10][...,1])
    mp3_audios_no_harmonics = [librosa.istft(mp3_spec) for mp3_spec in mp3_spectrograms]
    harmonic_spectrograms = harmonic_mp3[0:10][...,0] * np.exp(1j * harmonic_mp3[0:10][...,1])
    harmonic_audios = [librosa.istft(harm_spec) for harm_spec in harmonic_spectrograms]

    num_samples = 11
    plt.figure(figsize=(15, 5 * num_samples))  # Adjust size as necessary
    i = 1
    for mp3_audio, harmonic_audio in zip(mp3_audios_no_harmonics, harmonic_audios):
        # Original mp3 without harmonics
        plt.subplot(num_samples, 3, 3 * i + 1)
        D_original = librosa.amplitude_to_db(np.abs(librosa.stft(mp3_audio)), ref=np.max)
        plt.imshow(D_original, aspect='auto', origin='lower')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f"Original mp3 Spectrogram {i+1}")

        # mp3 with added harmonics
        plt.subplot(num_samples, 3, 3 * i + 2)
        D_harmonic = librosa.amplitude_to_db(np.abs(librosa.stft(harmonic_audio)), ref=np.max)
        plt.imshow(D_harmonic, aspect='auto', origin='lower')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f"Harmonic mp3 Spectrogram {i+1}")

        # Difference between the two
        plt.subplot(num_samples, 3, 3 * i + 3)
        D_diff = D_harmonic - D_original
        plt.imshow(D_diff, aspect='auto', origin='lower')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f"Difference Spectrogram {i+1}")

        i += 1

    plt.tight_layout()
    plt.show()
    
    for data in combined_mp3[0:3]:
        plot_phase(data[..., 1], sr=44100, hop_length=512)

    for data in harmonic_mp3[0:3]:
        plot_phase(data[..., 1], sr=44100, hop_length=512)

def polyfit_freq(combined_mp3, visualize=False):
    # 4. Use polyfit method to extend the transients
    harmonic_mp3 = extrapolate_frequency_content(combined_mp3)
    if visualize:
        visualize_polyfit(combined_mp3, harmonic_mp3)
    return harmonic_mp3

def split_and_normalize(harmonic_mp3):
    mag, phase, min, max = normalize_.normalize_data(harmonic_mp3)
    harmonic_mp3[..., 0] = mag
    harmonic_mp3[..., 1] = phase
    return min, max

def model_predict(harmonic_mp3, model_filepath):
    # 5. Plug in to model and get output
    with tf.keras.saving.custom_object_scope(custom_objects):
        print("loading model...")
        model = tf.keras.models.load_model(model_filepath, custom_objects=custom_objects, safe_mode=False) # safe_mode=False
                                                                                                           # bc model uses Lambda layers
        model.summary()
        # print(harmonic_mp3.shape)
        print("model.predict() -- can take a while...")
        pred = model.predict(harmonic_mp3, batch_size=1, verbose=1) # batch_size=1 bc of weirdness with tf BatchNormalization(), as
                                                                    # I trained with batch size of 1 for memory reasons and lo and behold
                                                                    # I shot myself in the foot bc the learned BN parameters do not
                                                                    # generalize to batch sizes more than 1... (sorry)
        # print(pred.shape)
        return pred

def polyfit_and_predict(combined_mp3, model_filepath, visualize=False):
    print("polyfitting frequencies")
    harmonic_mp3 = polyfit_freq(combined_mp3, visualize=visualize)
    print("splitting and normalizing")
    min, max = split_and_normalize(harmonic_mp3)
    print("predicting with model")
    pred = model_predict(harmonic_mp3, model_filepath)
    print("denormalizing")
    pred_mag, pred_phase = normalize_.denormalize_data(pred[..., 0], pred[..., 1], min, max)
    return pred_mag, pred_phase