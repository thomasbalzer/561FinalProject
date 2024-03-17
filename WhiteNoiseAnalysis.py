import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, fftfreq  # Import FFT and FFT frequency functions from scipy

def plot_frequency_response(data, fs):
    # Perform FFT for both original and filtered data
    if data.ndim > 1:
        data = data[:, 0]  # Use the first channel if it's stereo

    n = len(data)
    n_half = n//2  # Use n_half for one-sided spectrum
    T = 1.0 / fs

    yf_original = fft(data)  # Use scipy's FFT
    xf = fftfreq(n, d=T)[:n_half]  # Use scipy's FFT frequency function
    
    # Convert amplitude to dB
    yf_original_db = 20 * np.log10(2.0/n * np.abs(yf_original[:n_half]))
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(xf, yf_original_db, label='Original')
    plt.title('Frequency response comparison')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.legend()
    plt.grid()
    plt.show()

def main():
    # Settings
    fs, data = wavfile.read('whitenoise.wav')  # load the file

    # Check if measurement.wav file exists
    if os.path.isfile('measurement.wav'):
        fs_measurement, data_measurement = wavfile.read('measurement.wav')
        plot_frequency_response(data_measurement, fs_measurement)

    # Plot the frequency response of the original and the slightly adjusted signal
    plot_frequency_response(data, fs)

if __name__ == "__main__":
    main()
