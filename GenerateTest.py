import numpy as np
import scipy.signal as signal
import soundfile as sf
import matplotlib.pyplot as plt

def design_bandpass_filter(lowcut, highcut, fs, numtaps=101):
    """Design a bandpass filter using the FIR window method."""
    return signal.firwin(numtaps, [lowcut, highcut], pass_zero=False, fs=fs, window='hamming')

def apply_filter(data, filter_coefficients, gain=1.0):
    """Apply a filter to the data and scale the output by the specified gain."""
    filtered_data = signal.lfilter(filter_coefficients, [1.0], data)
    return filtered_data * gain

def fft_analysis(data, fs, n_fft=8192):
    """Compute the FFT of the data and return frequency bins and magnitude spectrum."""
    fft_data = np.fft.rfft(data, n_fft)
    freq = np.fft.rfftfreq(n_fft, 1/fs)
    magnitude = np.abs(fft_data)
    return freq, magnitude

def analyze_frequency_range(data, fs, threshold=0.01, n_fft=8192):
    """Analyze the audio to find the frequency range with significant energy."""
    freq, magnitude = fft_analysis(data, fs, n_fft)
    significant = magnitude > (np.max(magnitude) * threshold)
    min_freq = np.min(freq[significant]) if np.any(significant) else 20
    max_freq = np.max(freq[significant]) if np.any(significant) else fs / 2 - 1
    min_freq = max(min_freq, 1)
    max_freq = min(max_freq, fs / 2 - 1)
    return min_freq, max_freq

def main():
    # Load existing white noise
    filename = 'wavfiles/whitenoise.wav'
    data, fs = sf.read(filename)  # Read file and automatically get fs

    # Calculate the frequency bands ensuring they are within valid range
    nyquist = fs / 2
    num_filters = 3
    low_freqs, high_freqs = analyze_frequency_range(data, fs)
    bands = np.linspace(low_freqs, high_freqs, num_filters + 1)
    band_pairs = [(bands[i], bands[i + 1]) for i in range(len(bands) - 1)]
    gains = [0.5, 1, 0.8]  # Adjustable gains for each filter band

    # Design filters and apply them with respective gains
    filters = [design_bandpass_filter(low, high, fs) for low, high in band_pairs]
    filtered_signal = np.zeros_like(data)
    for bpf, gain in zip(filters, gains):
        filtered_signal += apply_filter(data, bpf, gain)

    # Save the filtered signal as a WAV file
    output_file = 'wavfiles/measurement.wav'
    sf.write(output_file, filtered_signal, fs)

    # Perform FFT analysis
    freq, magnitude = fft_analysis(filtered_signal, fs)

    # Plotting FFT of the measurement file
    plt.figure(figsize=(10, 5))
    plt.title('FFT of Filtered Measurement')
    plt.plot(freq, magnitude)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.xlim(0, nyquist)  # Limit x-axis to Nyquist frequency
    plt.show()

    print(f"Generated file '{output_file}' with filtered white noise, using adjustable gains.")

if __name__ == "__main__":
    main()
