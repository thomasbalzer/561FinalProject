import numpy as np
import soundfile as sf
import sounddevice as sd
import csv
import scipy.signal as signal
import os

def load_filters(csv_file):
    filters = []
    gains = []
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            gain = float(row[1])
            coefficients = np.array(row[2:], dtype=float)
            filters.append(coefficients)
            gains.append(gain)
    return filters, gains

def apply_filters(data, filters, gains):
    filtered_data = np.zeros_like(data)
    for b, gain in zip(filters, gains):
        filtered_output = signal.lfilter(b * gain, [1.0], data)
        filtered_data += filtered_output
    return filtered_data

def play_audio(file_path, filters, gains):
    data, fs = sf.read(file_path)
    if data.ndim > 1:
        data = np.mean(data, axis=1)  # Convert to mono if stereo

    filtered_data = apply_filters(data, filters, gains)

    # Ensuring the audio is in two channels
    stereo_data = np.column_stack((filtered_data, filtered_data))

    # Play audio
    sd.play(stereo_data, fs)
    sd.wait()

def main():
    directory = 'songs'
    csv_file = 'filter_coefficients.csv'

    # Load filters and gains
    filters, gains = load_filters(csv_file)

    # List all songs in the directory
    songs = [f for f in os.listdir(directory) if f.endswith('.wav')]
    if not songs:
        print("No songs found in the directory.")
        return

    # Just playing the first song for demonstration
    song_path = os.path.join(directory, songs[0])
    print(f"Playing: {song_path}")
    
    play_audio(song_path, filters, gains)

if __name__ == "__main__":
    main()
