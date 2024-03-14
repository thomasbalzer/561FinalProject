import numpy as np
from scipy.io import wavfile

import matplotlib.pyplot as plt

# Load the audio file
sample_rate, data = wavfile.read('whitenoise.wav')

# Perform FFT
fft_data = np.fft.fft(data)

# Calculate the frequency bins
freq_bins = np.fft.fftfreq(len(data), 1/sample_rate)

# Find the indices corresponding to the audible spectrum
audible_start = 20  # Minimum audible frequency (20 Hz)
audible_end = 20000  # Maximum audible frequency (20,000 Hz)
audible_indices = np.where((freq_bins >= audible_start) & (freq_bins <= audible_end))

# Convert amplitude to decibel scale
amplitude_dB = 20 * np.log10(np.abs(fft_data[audible_indices]))

#Test
# Plot the FFT in decibel scale
plt.plot(freq_bins[audible_indices], amplitude_dB)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (dB)')
plt.title('FFT of whitenoise.wav (Audible Spectrum in dB)')
plt.show()