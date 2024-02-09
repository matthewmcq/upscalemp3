# upscalemp3

Converts an mp3 (lossy) file back into its uncompressed wav counterpart based on a generative AI model built with tensorflow.

## Control Flow Outline:

1. mp3 and corresponding wav audio files are broken into overlapping 1 second segements for training.
2. These audio segments are transformed into their spectral composition using the (librosa) STFT and interleaved to support stereo.
3. A polynomial regression is run on the magnitudes of each sample for every time step to roughly recover lost higher frequencies
4. These augmented mp3 spectrograms are passed along with their corresponding wav spectrograms into a UNet-style neural network with residual
   encoder blocks (ResUNet) to clean up the polynomial regression and add more precision to magnitudes. Note: Phases are omitted from the model, as the Griffin-Lim algorithm does a better job (and because it cuts the size of the model in half).
5. After training, the model predicts the missing spectrogram data from input mp3 audio segments and returns the ISTFT of the spectrogram
   encoded as a wav file at 44.1kHz.
6. The overlapping interleaved segments are combined using OLA and a hanning window (this seems to be producing some artifacts based on the hard 1s splits in the original mp3, so I might experiment with using zero-cross cutting to minimize weirdness with the spectrograms) 
7. Finally, each channel is run through a slightly modified Griffin-Lim algorithm to rebuild the correct phases. niter=200 by default, but this can  be super slow. That said, anything under 100 sounds noticible worse.
8. The L and R channels are combined, and the output is encoded as a 24-bit PCM .wav file that gets written to the designated output filepath as "upscaled_mp3.wav"
