import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, lfilter, freqz

# Function to design a Butterworth Low-Pass Filter and apply it
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    # Design Butterworth low-pass filter
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # Apply the filter to the data
    filtered_data = lfilter(b, a, data)
    return filtered_data

# Load the wav file
fs, recorded_noise = wavfile.read('whitenoise.wav')

# Ensure the audio data is in floating point format for processing
# Convert it if necessary
if recorded_noise.dtype != np.float32:
    # Normalize 16-bit WAV file data
    recorded_noise = recorded_noise / np.max(np.abs(recorded_noise))

# Design and apply the low-pass filter
cutoff_frequency = 5000  # Set your cutoff frequency (Hz)
filtered_noise = butter_lowpass_filter(recorded_noise, cutoff_frequency, fs)

# Perform FFT on the original and filtered signals
fft_magnitude_original = np.abs(np.fft.rfft(recorded_noise))
fft_magnitude_filtered = np.abs(np.fft.rfft(filtered_noise))

# Frequency bins for plotting
frequencies = np.fft.rfftfreq(len(recorded_noise), 1/fs)

# Plot FFT of original and filtered recorded noise
plt.figure(figsize=(15, 6))

# Original signal FFT
plt.plot(frequencies, fft_magnitude_original, label='Original Noise')

# Filtered signal FFT
plt.plot(frequencies, fft_magnitude_filtered, label='Filtered Noise')

plt.title('FFT of Recorded Noise and Filtered Noise')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.legend()
plt.grid(True)
plt.show()
