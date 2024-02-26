import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def plot_frequency_response(original_data, filtered_data, fs):
    # Perform FFT for both original and filtered data
    n = len(original_data)
    T = 1.0 / fs
    yf_original = np.fft.fft(original_data)
    yf_filtered = np.fft.fft(filtered_data)
    xf = np.linspace(0.0, 1.0/(2.0*T), n//2)
    
    # Convert amplitude to dB
    yf_original_db = 20 * np.log10(2.0/n * np.abs(yf_original[:n//2]))
    yf_filtered_db = 20 * np.log10(2.0/n * np.abs(yf_filtered[:n//2]))
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(xf, yf_original_db, label='Original')
    plt.plot(xf, yf_filtered_db, label='Filtered', linestyle='--')
    plt.title('Frequency response comparison')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.legend()
    plt.grid()
    plt.show()

def main():
    # Settings
    fs, data = wavfile.read('whitenoise.wav')  # load the file

    # Ensure the data is in the correct shape
    if data.ndim > 1:
        data = data[:, 0]  # Use the first channel if it's stereo

    # Slightly adjust the data to simulate room effect (Example adjustment)
    # This is a placeholder for a subtle effect. You may need a more sophisticated approach for realistic simulation.
    filtered_data = np.copy(data) * 0.99  # Example subtle change

    # Plot the frequency response of the original and the slightly adjusted signal
    plot_frequency_response(data, filtered_data, fs)

if __name__ == "__main__":
    main()
