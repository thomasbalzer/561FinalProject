import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, fftfreq, ifft
from scipy.signal import firwin, lfilter, savgol_filter, hamming

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

def inverse_filter_design_with_fitted_curve(fs, signal, numtaps, regularization=0.01, window_length=101, polyorder=3, fit_order=5):
    fft_signal = fft(signal)
    magnitude = np.abs(fft_signal)
    phase = np.angle(fft_signal)
    
    normalized_magnitude = magnitude / np.max(magnitude)
    smoothed_magnitude = savgol_filter(normalized_magnitude, window_length, polyorder)

    inverse_magnitude = 1 / (smoothed_magnitude + regularization)
    freqs = fftfreq(len(signal), 1/fs)
    coeffs = np.polyfit(freqs[:len(signal)//2], inverse_magnitude[:len(signal)//2], fit_order)
    fitted_inverse = np.polyval(coeffs, freqs[:len(signal)//2])
    fitted_full = np.concatenate([fitted_inverse, fitted_inverse[::-1]])

    inverse_fft = fitted_full * np.exp(1j * phase[:len(fitted_full)])
    taps_complex = ifft(inverse_fft)[:numtaps]
    taps_complex *= hamming(numtaps)
    
    return taps_complex, magnitude, smoothed_magnitude, fitted_inverse

def plot_results(fs, original_signal, corrected_signal, smoothed_magnitude, fitted_inverse):
    # Frequency domain for plots
    freqs = fftfreq(len(original_signal), 1/fs)[:len(original_signal)//2]

    # Original Signal FFT Magnitude
    original_fft_magnitude = np.abs(fft(original_signal))[:len(original_signal)//2]

    # Corrected Signal FFT Magnitude
    corrected_fft_magnitude = np.abs(fft(corrected_signal))[:len(corrected_signal)//2]

    plt.figure(figsize=(15, 12))

    # Plot 1: Smoothed and Fitted Inverse Magnitude
    plt.subplot(3, 1, 1)
    plt.plot(freqs, smoothed_magnitude[:len(freqs)], label='Smoothed Magnitude', linestyle='--')
    plt.plot(freqs, fitted_inverse, label='Fitted Inverse Magnitude')
    plt.title('Smoothed and Fitted Inverse Magnitude')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid(True)

    # Plot 2: Original FFT Magnitude
    plt.subplot(3, 1, 2)
    plt.plot(freqs, original_fft_magnitude, label='Original FFT Magnitude')
    plt.title('Original FFT Magnitude')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid(True)

    # Plot 3: Corrected FFT Magnitude
    plt.subplot(3, 1, 3)
    plt.plot(freqs, corrected_fft_magnitude, label='Corrected FFT Magnitude')
    plt.title('Corrected FFT Magnitude')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def main():
    fs, data = wavfile.read('wavfiles/measurement.wav')
    if data.ndim > 1:
        data = data[:, 0]

    numtaps = 1001
    third_fs = fs // 6

    # Process the signal
    lowpass_data, _ = apply_fir_filter(data, fs, numtaps, 'lowpass', third_fs)
    bandpass_data, _ = apply_fir_filter(data, fs, numtaps, 'bandpass', [third_fs, third_fs*2])
    highpass_data, _ = apply_fir_filter(data, fs, numtaps, 'highpass', third_fs*2)
    composite_signal = lowpass_data * 0.6 + bandpass_data * 0.4 + highpass_data * 0.5

    # Design the correction filter
    correction_taps, original_fft, smoothed_fft, fitted_inverse = inverse_filter_design_with_fitted_curve(fs, composite_signal, numtaps)

    # Apply the correction filter
    corrected_signal = lfilter(correction_taps.real, 1.0, composite_signal)

    # Visualize the results
    plot_results(fs, composite_signal, corrected_signal, smoothed_fft, fitted_inverse)

if __name__ == "__main__":
    main()
