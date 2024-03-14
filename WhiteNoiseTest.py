import numpy as np
from scipy.io import wavfile

import matplotlib.pyplot as plt

# Step 1: Read the WAV file
sampling_rate, data = wavfile.read('whitenoise.wav')

# Step 2: Compute the FFT
fft_result = np.fft.fft(data)
fft_magnitude = np.abs(fft_result)

# Step 3: Calculate Frequency Bins
n = len(data)
frequency_bins = np.fft.fftfreq(n, d=1/sampling_rate)

# Step 4: Normalize the FFT Magnitude
normalized_fft_magnitude = fft_magnitude / np.max(fft_magnitude)

# Step 5: Plot the Frequency Response
plt.figure(figsize=(10, 6))
plt.plot(frequency_bins[:n // 2], normalized_fft_magnitude[:n // 2])  # Plot only the positive frequencies
plt.title('Frequency Response of White Noise')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()
