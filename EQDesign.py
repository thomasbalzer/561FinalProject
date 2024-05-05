import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import soundfile as sf
import csv

def fft_analysis(data, fs, n_fft=4096):
    fft_data = np.fft.rfft(data, n_fft)
    freq = np.fft.rfftfreq(n_fft, 1/fs)
    magnitude = np.abs(fft_data)
    return freq, magnitude

def analyze_frequency_range(data, fs):
    # Start from a small value above 0 Hz, up to 22 kHz, not exceeding fs/2
    min_freq = 1  # Start slightly above 0 Hz to avoid design issues
    max_freq = min(22000, fs / 2 - 1)  # Ensure it does not exceed Nyquist frequency
    return min_freq, max_freq

def design_filters(fs, min_freq, max_freq, num_filters, numtaps=101):
    frequencies = np.linspace(min_freq, max_freq, num_filters + 1)
    filters = []
    for i in range(num_filters):
        # Create bandpass filters between consecutive frequency points
        b = signal.firwin(numtaps, [frequencies[i], frequencies[i + 1]], pass_zero=False, fs=fs, window='hamming')
        filters.append(b)
    return filters

def apply_filters(data, filters, gains):
    output = np.zeros_like(data)
    for b, gain in zip(filters, gains):
        filtered_data = signal.lfilter(b * gain, [1.0], data)
        output += filtered_data
    return output

def save_filters_to_file(filters, adjustment_factors, filename='filter_coefficients.csv'):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Filter Number', 'Adjustment Factor', 'Filter Coefficients'])
        for i, (filter_coeffs, factor) in enumerate(zip(filters, adjustment_factors)):
            capped_factor = min(factor, 2)  # Cap to prevent excessive gain
            writer.writerow([i + 1, capped_factor] + list(filter_coeffs))

def main():
    filename = 'wavfiles/measurement.wav'
    data, fs = sf.read(filename)
    if data.ndim == 2:
        data = np.mean(data, axis=1)

    min_freq, max_freq = analyze_frequency_range(data, fs)
    print(f"Frequency range with significant energy: {min_freq} Hz to {max_freq} Hz")

    num_filters = 10
    num_taps = 101
    filters = design_filters(fs, min_freq, max_freq, num_filters, num_taps)

    band_energies = []
    for b in filters:
        filtered_data = signal.lfilter(b, [1.0], data)
        _, filtered_magnitude = fft_analysis(filtered_data, fs)
        band_energy = np.sum(filtered_magnitude ** 2)
        band_energies.append(band_energy)

    max_energy = max(band_energies)
    adjustment_factors = [min(np.sqrt(max_energy / e), 2) if e > 0 else 1 for e in band_energies]

    save_filters_to_file(filters, adjustment_factors)

    filtered_data = apply_filters(data, filters, adjustment_factors)

    plt.figure(figsize=(8, 6), dpi=80)
    plt.subplots_adjust(hspace=0.5, wspace=0.4)

    plt.subplot(2, 2, 1)
    freq, original_magnitude = fft_analysis(data, fs)
    plt.semilogy(freq, original_magnitude, label='Original Magnitude')
    plt.title('FFT of Original Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    freq, filtered_magnitude = fft_analysis(filtered_data, fs)
    plt.semilogy(freq, filtered_magnitude, label='Filtered Magnitude')
    plt.title('FFT of Filtered Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    for i, b in enumerate(filters):
        w, h = signal.freqz(b * adjustment_factors[i], worN=8000, fs=fs)
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
