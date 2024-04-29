import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import soundfile as sf

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

def design_filters(fs, min_freq, max_freq, num_filters, numtaps=101):
    """Design bandpass filters within specified frequency bounds."""
    frequencies = np.linspace(min_freq, max_freq, num_filters + 1)
    filters = []
    for i in range(num_filters):
        b = signal.firwin(numtaps, [frequencies[i], frequencies[i + 1]], pass_zero=False, fs=fs, window='hamming')
        filters.append(b)
    return filters

def apply_filters(data, filters, gains):
    """Apply the designed filters to the data with adjusted gains and aggregate the outputs."""
    output = np.zeros_like(data)
    for b, gain in zip(filters, gains):
        filtered_data = signal.lfilter(b * gain, [1.0], data)  # Apply adjusted gain
        output += filtered_data
    return output

def main():
    filename = 'wavfiles/measurement.wav'
    data, fs = sf.read(filename)

    # Analyze frequency range of the audio file
    min_freq, max_freq = analyze_frequency_range(data, fs)
    print(f"Frequency range with significant energy: {min_freq} Hz to {max_freq} Hz")

    # Design filters based on analyzed frequency range
    num_filters = 10
    filters = design_filters(fs, min_freq, max_freq, num_filters)

    # FFT analysis of the original data to use in plotting
    freq, original_magnitude = fft_analysis(data, fs)
    
    # Calculate adjustment factors based on energy in each band
    adjustment_factors = []
    total_original_energy = np.sum(original_magnitude ** 2)
    print("Energy in each band:")
    for i, b in enumerate(filters):
        w, h = signal.freqz(b, worN=len(original_magnitude), fs=fs)
        filter_response = np.abs(h)
        band_energy = np.sum((filter_response ** 2) * original_magnitude)
        band_energy_proportion = band_energy / total_original_energy
        adjustment_factors.append(1 / np.sqrt(band_energy_proportion))  # Normalize to make quieter bands louder
        print(f"Band {i + 1}: Energy = {band_energy:.2f}, Proportion = {band_energy_proportion:.2f}")

    # Debug statement to print calculated gains
    print("Calculated gains for each filter:")
    for i, gain in enumerate(adjustment_factors):
        print(f"Filter {i + 1}: Gain = {gain:.2f}")

    # Adjust gains based on calculated factors
    adjusted_gains = [f for f in adjustment_factors]  # Adjusted gains based on normalized filter responses

    # Apply filters with adjusted gains
    filtered_data = apply_filters(data, filters, adjusted_gains)

    # FFT analysis after filtering
    _, filtered_magnitude = fft_analysis(filtered_data, fs)

    # Plotting results
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.semilogy(freq, original_magnitude, label='Original Magnitude')
    plt.title('FFT of Original Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.semilogy(freq, filtered_magnitude, label='Filtered Magnitude')
    plt.title('FFT of Filtered Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    for i, b in enumerate(filters):
        w, h = signal.freqz(b * adjusted_gains[i], worN=8000, fs=fs)  # Apply adjusted gains
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label=f'Filter {i + 1}')
    plt.title('Frequency Responses of Each Bandpass Filter')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Gain')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
