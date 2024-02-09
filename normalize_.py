import librosa
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import gc
import soundfile as sf
import tqdm



def compute_min_max(arr, is_phase=False):
    """Compute min and max for the array."""
    if not is_phase:
        arr = np.log1p(abs(arr))
    min_val = np.min(arr)
    max_val = np.max(arr)
    return min_val, max_val

def denormalize_with_given_min_max(arr, min_val, max_val, is_phase=False):
    """Denormalize the array using given min and max and return the denormalized array."""
    arr = arr * (max_val - min_val) + min_val
    if not is_phase:
        arr = np.expm1(arr)  # inverse of log1p
    return arr

def denormalize_data(pred_mag_norm, pred_phase_norm, mag_min_val, mag_max_val):
    # Denormalize magnitude
    pred_mag = denormalize_with_given_min_max(pred_mag_norm, mag_min_val, mag_max_val)
    
    pred_phase = denormalize_phase(pred_phase_norm)
    return pred_mag, pred_phase

def normalize_with_given_min_max(arr, min_val, max_val, is_phase=False):
    """Normalize the array using given min and max and return the normalized array."""
    if not is_phase:
        arr = np.log1p(np.abs(arr))
    return (arr - min_val) / (max_val - min_val)

def normalize_phase(arr):
    """Normalize phase values to the range [0, 1] from [-π, π]."""
    return (arr + np.pi) / (2 * np.pi)

def denormalize_mag(arr):
  return np.exp(arr) - 1

def denormalize_phase(arr):
    """Denormalize phase values from the range [0, 1] to [-π, π]."""
    return arr * 2 * np.pi - np.pi

def normalize_data(combined_mp3):
    pred_min = np.min(combined_mp3[..., 0])
    pred_max = np.max(combined_mp3[..., 0])

    # Normalize based on global min and max
    pred_mag = normalize_with_given_min_max(combined_mp3[..., 0], pred_min, pred_max)

    pred_phase = normalize_phase(combined_mp3[..., 1])

    return pred_mag, pred_phase, pred_min, pred_max