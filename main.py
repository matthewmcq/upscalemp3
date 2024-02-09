# Control Flow:

# Inputs: mp3 file descriptor, destination file descriptor
# Outputs: wav file at destination file descriptor

# PREPROCESSING:
# 0. Use librosa to resample mp3 file as a 44.1kHz wav file
# 1.  (if stereo -> do everything twice, once for each channel)
#  len > 1.5s |-> Split mp3 into 1 second chunks with 0.5 second overlap (if \in [1, 1.5]s, pad with zeros at beginning and end to 1.5 seconds)
#  len < 1s   |-> Pad with zeros at beginning and end to 1 second
#  len = 1s   |-> Do nothing extra
#  len = 0s   |-> Error
# 2. For each chunk, compute the STFT with hop len 512 and nfft 2048
# 3. Pad magnitude and phase to be of shape 1025, 87

# MODEL:
# 4. Use polyfit method to extend the transients
# 5. Plug in to model and get output

# POSTPROCESSING:
# 6. Stitch the spectrograms together along the time axis
# 7. Run ISTFT with modified griffin lim algorithm on the stitched spectrogram to get the output audio
#    |-> Modify griffin lim algorithm to use/start with the phase from the output of the model (should try with and without)
# 8. Write output audio to file

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import soundfile as sf
import preprocessing
import predict
import postprocessing
import sys



# TODO - noise reduction?
# Identify artifats (perhaps with std dev for magnitude coefficients, and ignore normalization for those)
# Item - by item normalization (deal with batchnorm) or use moving average/minmax from original mp3

def separate_channels(pred_mag, pred_phase):
    # Assuming even indices are left channel and odd indices are right channel
    pred_mag_left = pred_mag[::2]    # Slices out the left channel magnitudes
    pred_phase_left = pred_phase[::2]  # Slices out the left channel phases

    pred_mag_right = pred_mag[1::2]  # Slices out the right channel magnitudes
    pred_phase_right = pred_phase[1::2]  # Slices out the right channel phases

    return pred_mag_left, pred_phase_left, pred_mag_right, pred_phase_right

if __name__ == "__main__":

    MODEL_FILEPATH = 'models/ResUNet_LSTM_small3GB.keras'
    
    # command line args 1 and 2 are the input and output filepaths
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    # preprocessing
    print("Preprocessing")
    combined_mp3 = preprocessing.preprocess(input_path)

    # predict
    print("Predicting")
    mags, phases = predict.polyfit_and_predict(combined_mp3, MODEL_FILEPATH)

    # Separate the predictions into left and right channels
    pred_mag_left, pred_phase_left, pred_mag_right, pred_phase_right = separate_channels(mags, phases)

    # postprocessing
    print("Postprocessing")
    output_audio_gl_left = postprocessing.postprocess(pred_mag_left, pred_phase_left, visualize=True)
    output_audio_gl_right = postprocessing.postprocess(pred_mag_right, pred_phase_right, visualize=True)

    # Combine the left and right channels into stereo
    output_audio_gl_stereo = np.vstack((output_audio_gl_left, output_audio_gl_right)).T
    
    # Save both as wav files to output_path
    print("Saving")
    sf.write(output_path + 'upscaled_mp3.wav', output_audio_gl_stereo, 44100, "PCM_24")

    

