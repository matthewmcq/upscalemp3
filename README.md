# upscalemp3

Converts an mp3 (lossy) file back into its uncompressed wav counterpart based on a generative AI model built with tensorflow.

## upscalemp3_training_and_preprocessing.ipynb:

1. mp3 and corresponding wav audio files are broken into overlapping 1 second segements for training.
2. These audio segments are transformed into their spectral composition using the (librosa) STFT
3. A polynomial regression is run on the magnitudes of each sample for every time step to roughly recover lost higher frequencies
4. These augmented mp3 spectrograms are passed along with their corresponding wav spectrograms into a UNet-style neural network with residual
   encoder blocks (U-ResNet) to clean up the polynomial regression and add more precision to magnitudes
5. After training, the model predicts the missing spectrogram data from input mp3 audio segments and returns the ISTFT of the spectrogram
   encoded as a wav file at 44.1kHz.

## TODO:

- Need a way to stitch arbitrarily-sized mp3 files back together after running the model on 1s segments.
- Compensate for slight phase differences with an augmented Griffin-Lim algorithm.
