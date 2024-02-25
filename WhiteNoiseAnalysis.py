import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, lfilter

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def plot_frequency_response(data, fs):
    # Perform FFT
    n = len(data)
    T = 1.0 / fs
    yf = np.fft.fft(data)
    xf = np.linspace(0.0, 1.0/(2.0*T), n//2)
    
    # Convert to dB
    yf_db = 20 * np.log10(2.0/n * np.abs(yf[:n//2]))
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(xf, yf_db)
    plt.title('Frequency response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.grid()
    plt.show()

def main():
    # Settings
    order = 6
    fs, data = wavfile.read('whitenoise.wav')  # load the file
    cutoff = 1000  # Cutoff frequency of the filter in Hz, adjust as needed

    # Ensure the data is in the correct shape
    if data.ndim > 1:
        data = data[:, 0]  # Use the first channel if it's stereo

    # Apply filter
    filtered_data = butter_lowpass_filter(data, cutoff, fs, order)

    # Plot the frequency response of the filtered signal
    plot_frequency_response(filtered_data, fs)

if __name__ == "__main__":
    main()
