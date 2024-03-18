import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, fftfreq, ifft
from scipy.signal import firwin, lfilter, convolve, savgol_filter

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

def savgol_smoothing_filter(signal, window_length=51, polyorder=3):
    if window_length % 2 == 0:  # Ensure window_length is odd
        window_length += 1
    return savgol_filter(signal, window_length, polyorder)

def inverse_filter_design(fs, signal, numtaps, regularization=1e-6):
    fft_signal = np.abs(fft(signal))/np.max(np.abs(fft(signal)))
    freqs = fftfreq(len(signal), 1/fs)
    
    inverse_fft = 1 / (fft_signal + regularization)
    max_val = np.max(np.abs(inverse_fft))
    inverse_fft[np.abs(inverse_fft) > max_val] = max_val
    
    taps = np.real(ifft(inverse_fft))[:numtaps]
    taps *= np.hamming(numtaps)
    
    return taps

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

def main():
    fs, data = wavfile.read('whitenoise.wav')  # Assume a path to your white noise file
    if data.ndim > 1:
        data = data[:, 0]  # Use the first channel if stereo

    numtaps = 101
    third_fs = fs // 6

    # Apply filters
    lowpass_data, _ = apply_fir_filter(data, fs, numtaps, 'lowpass', third_fs)
    bandpass_data, _ = apply_fir_filter(data, fs, numtaps, 'bandpass', [third_fs, third_fs*2])
    highpass_data, _ = apply_fir_filter(data, fs, numtaps, 'highpass', third_fs*2)
    
    # Apply magnitude adjustments
    lowpass_adjusted = lowpass_data * 0.7
    bandpass_adjusted = bandpass_data * 0.4
    highpass_adjusted = highpass_data * 0.5
    
    composite_signal = lowpass_adjusted + bandpass_adjusted + highpass_adjusted
    
    smoothed_composite_signal = savgol_smoothing_filter(composite_signal, 5, 4)
    
    correction_taps = inverse_filter_design(fs, smoothed_composite_signal, numtaps, regularization=0.001)
    
    corrected_signal = lfilter(correction_taps, 1.0, smoothed_composite_signal)
    
    signals = [smoothed_composite_signal, corrected_signal, correction_taps]
    titles = ['Smoothed Composite Signal', 'Corrected Signal', 'Correction Filter']
    plot_frequency_responses(fs, signals, titles)

main()  # Uncomment this line to run the script in an appropriate environment.
