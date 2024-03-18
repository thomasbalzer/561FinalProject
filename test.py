import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
from scipy.signal import firwin, lfilter
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

def apply_fir_filter(data, fs, numtaps, filter_type, cutoffs):
    if filter_type == 'lowpass':
        taps = firwin(numtaps, cutoffs, fs=fs, pass_zero='lowpass')
    elif filter_type == 'highpass':
        taps = firwin(numtaps, cutoffs, fs=fs, pass_zero='highpass')
    elif filter_type == 'bandpass':
        taps = firwin(numtaps, [cutoffs, cutoffs*2], fs=fs, pass_zero=False)
    else:
        raise ValueError("filter_type must be 'lowpass', 'highpass', or 'bandpass'")
    filtered_data = lfilter(taps, 1.0, data)
    return filtered_data, taps

def linear_function(f, m, b):
    return m * f + b

def main():
    # Load the white noise file
    fs, data = wavfile.read('whitenoise.wav')  
    if data.ndim > 1:
        data = data[:, 0]  # Use the first channel if stereo

    numtaps = 101
    third_fs = fs // 6

    # Apply filters
    lowpass_data, _ = apply_fir_filter(data, fs, numtaps, 'lowpass', third_fs)
    bandpass_data, _ = apply_fir_filter(data, fs, numtaps, 'bandpass', third_fs)
    highpass_data, _ = apply_fir_filter(data, fs, numtaps, 'highpass', third_fs*2)

    # Apply magnitude adjustments
    lowpass_adjusted = lowpass_data * 0.7
    bandpass_adjusted = bandpass_data * 0.4
    highpass_adjusted = highpass_data * 0.5

    # Create the composite signal
    composite_signal = lowpass_adjusted + bandpass_adjusted + highpass_adjusted

    # FFT of the composite signal
    N = len(composite_signal)
    f = fftfreq(N, 1/fs)[:N//2]
    fft_signal = np.abs(fft(composite_signal))[:N//2]

    # Fit a line to the magnitude spectrum of the composite signal
    params, _ = curve_fit(linear_function, f, fft_signal, p0=[1, 1])
    slope, intercept = params

    # Generate the fitted line for plotting
    fitted_line = linear_function(f, slope, intercept)

    # Plot original and fitted frequency response
    plt.figure(figsize=(12, 6))
    plt.plot(f, fft_signal, label='Original FFT Magnitude', alpha=0.7)
    plt.plot(f, fitted_line, label='Fitted Line', linestyle='--')
    plt.title('FFT Magnitude and Fitted Line')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.show()

# Ensure the path to your white noise file is correct before running this script.
main()  # Uncomment to run the script in your environment
