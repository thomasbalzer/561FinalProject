import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

# Load the WAV file
sample_rate, data = wav.read('wavfiles/measurement.wav')

# Calculate the length of the audio in seconds
duration = len(data) / sample_rate

# Perform Fourier Transform
n = len(data)
fourier_transform = np.fft.fft(data)
frequencies = np.fft.fftfreq(n, 1 / sample_rate)
magnitude_spectrum = np.abs(fourier_transform)

# Plot the energy/frequency relationship
plt.figure(figsize=(10, 6))
plt.plot(frequencies[:n//2], magnitude_spectrum[:n//2])
plt.title('Energy/Frequency Relationship')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Energy')
plt.grid(True)
plt.xlim(0, sample_rate / 2)
plt.ylim(0, np.max(magnitude_spectrum) * 1.1)
plt.show()
