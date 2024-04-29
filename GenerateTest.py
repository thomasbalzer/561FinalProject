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

def fft_analysis(data, fs):
    """Perform FFT analysis and return frequency bins and magnitude spectrum."""
    fft_data = np.fft.rfft(data)
    freq = np.fft.rfftfreq(len(data), 1/fs)
    magnitude = np.abs(fft_data)
    return freq, magnitude

def main():
    # Load existing white noise
    filename = 'wavfiles/whitenoise.wav'
    data, fs = sf.read(filename)  # Read file and automatically get fs

    # Calculate the frequency bands ensuring they are within valid range
    nyquist = fs / 2
    num_filters = 3
    low_freqs = np.linspace(20, nyquist - 100, num_filters + 1)[:-1]  # Starting at 20 to avoid 0Hz, ending a bit below Nyquist
    high_freqs = np.linspace(20, nyquist - 100, num_filters + 1)[1:]  # Ensuring the upper cutoffs do not cause an error
    bands = list(zip(low_freqs, high_freqs))
    gains = [0.1, 1, 1.8]  # Adjustable gains for each filter band

    # Design filters and apply them with respective gains
    filters = [design_bandpass_filter(low, high, fs) for low, high in bands]
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
