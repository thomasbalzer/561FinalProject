import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
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
    return filtered_data

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
    fs, data = wavfile.read('wavfiles\whitenoise.wav')  # Load the file
    if data.ndim > 1:
        data = data[:, 0]  # Use the first channel if it's stereo

    numtaps = 101  # Number of taps in the FIR filter
    third_fs = fs // 6  # One-third of Nyquist frequency

    # Apply filters
    lowpass_data = apply_fir_filter(data, fs, numtaps, 'lowpass', third_fs)
    bandpass_data = apply_fir_filter(data, fs, numtaps, 'bandpass', [third_fs, third_fs*2])
    highpass_data = apply_fir_filter(data, fs, numtaps, 'highpass', third_fs*2)
    
    # Apply magnitude adjustments
    lowpass_adjusted = lowpass_data * 0.6
    bandpass_adjusted = bandpass_data * 0.4
    highpass_adjusted = highpass_data * 0.5
    
    # Sum the adjusted signals to create a new composite signal
    composite_signal = lowpass_adjusted + bandpass_adjusted + highpass_adjusted
    
    # Plot the FFT of the signals
    signals = [
        data, 
        #lowpass_adjusted, 
        #bandpass_adjusted, 
        #highpass_adjusted, 
        composite_signal,
               ]
    titles = [
        'Original Data',
        #'Lowpass Filtered', 
        #'Bandpass Filtered', 
        #'Highpass Filtered', 
        'Composite Signal',
        ]
    plot_frequency_responses(fs, signals, titles)

main() # Uncomment for execution outside of this environment.
