import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, fftfreq, ifft
from scipy.signal import firwin, lfilter

def apply_fir_filter(data, fs, numtaps, filter_type, cutoffs):
    if filter_type == 'lowpass':
        taps = firwin(numtaps, cutoffs, fs=fs, pass_zero='lowpass')
    elif filter_type == 'highpass':
        taps = firwin(numtaps, cutoffs, fs=fs, pass_zero='highpass')
    elif filter_type == 'bandpass':
        taps = firwin(numtaps, cutoffs, fs=fs, pass_zero=False)
    else:
        raise ValueError("filter_type must be 'lowpass', 'highpass', or 'bandpass'")
    filtered_data = lfilter(taps, 1.0, data)
    return filtered_data, taps

def inverse_filter_design(fs, signal, numtaps, regularization=0.001):
    fft_signal = fft(signal)/max(fft(signal))
    inverse_fft = 1 / (np.abs(fft_signal) + regularization) * np.exp(1j * np.angle(fft_signal))

    # Window function to reduce artifacts
    taps_complex = ifft(inverse_fft)[:numtaps]
    taps_complex *= np.hamming(numtaps)
    
    return taps_complex

def plot_frequency_responses(fs, signals, titles):
    plt.figure(figsize=(12, 6))
    for signal, title in zip(signals, titles):
        N = len(signal)
        f = fftfreq(N, 1/fs)[:N//2]
        fft_signal = np.abs(fft(signal))[:N//2]
        plt.plot(f, fft_signal/np.max(fft_signal), label=title, alpha=0.7)
    
    plt.title('Frequency Responses')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.legend()
    plt.show()

def align_signals(reference_signal, signal_to_align, numtaps):
    # Estimate delay
    delay = numtaps // 2
    # Zero-padding to align signals
    aligned_signal = np.roll(signal_to_align, -delay)
    return aligned_signal

def main():
    fs, data = wavfile.read('wavfiles\whitenoise.wav')  # Path to your white noise file
    if data.ndim > 1:
        data = data[:, 0]  # Use the first channel if stereo

    numtaps = 201 
    third_fs = fs // 6

    # Apply filters
    lowpass_data, _ = apply_fir_filter(data, fs, numtaps, 'lowpass', third_fs)
    bandpass_data, _ = apply_fir_filter(data, fs, numtaps, 'bandpass', [third_fs, third_fs*2])
    highpass_data, _ = apply_fir_filter(data, fs, numtaps, 'highpass', third_fs*2)

    # Create composite signal
    composite_signal = lowpass_data * 0.6 + bandpass_data * 0.4 + highpass_data * 0.5

    # Design inverse filter
    correction_taps = inverse_filter_design(fs, composite_signal, numtaps)

    # Apply inverse filter
    corrected_signal = lfilter(correction_taps.real, [1.0], composite_signal)

    # Align corrected signal with the original composite signal
    aligned_corrected_signal = align_signals(composite_signal, corrected_signal, numtaps)

    # Plot responses
    signals = [composite_signal, aligned_corrected_signal, correction_taps.real]
    titles = ['Composite Signal', 'Aligned Corrected Signal', 'Correction Filter']
    plot_frequency_responses(fs, signals, titles)

# Run the main function
main()
