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

import librosa
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import gc
import soundfile as sf
import tqdm
import preprocessing
import predict
import postprocessing
import normalize
import sys

if __name__ == "__main__":

    MODEL_FILEPATH = 'models/model1'
    
    # command line args 1 and 2 are the input and split filepaths
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    # preprocessing
    combined_mp3 = preprocessing.preprocess(input_path)

    # predict
    prediction = predict.model_predict(combined_mp3, MODEL_FILEPATH)

    # postprocess
    gl_audio, normal_audio = postprocessing.postprocess(prediction, visualize=True)

    # Save both as wav files to output_path
    sf.write(output_path + 'gl.wav', gl_audio, 44100)
    sf.write(output_path + 'normal.wav', normal_audio, 44100)

    

