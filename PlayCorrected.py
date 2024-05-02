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
        for ch in range(data.shape[1]):  # Apply filters to each channel
            filtered_output = signal.lfilter(b * gain, [1.0], data[:, ch])
            filtered_data[:, ch] += filtered_output
    return filtered_data

def audio_stream_generator(file_path, block_size):
    with sf.SoundFile(file_path) as sf_file:
        while True:
            data = sf_file.read(block_size)
            if len(data) == 0:
                break
            yield data

def play_audio(file_path, filters, gains):
    block_size = 1024  # Block size for chunks
    data_generator = audio_stream_generator(file_path, block_size)

    def callback(outdata, frames, time, status):
        try:
            data = next(data_generator)
            if data.shape[1] < 2:
                data = np.column_stack((data, data))  # Ensure data has two channels
            filtered_data = apply_filters(data, filters, gains)
            outdata[:] = filtered_data
        except StopIteration:
            outdata.fill(0)  # Fill remaining buffer with zeros
            raise sd.CallbackStop

    # Get the sample rate from the file
    with sf.SoundFile(file_path) as sf_file:
        fs = sf_file.samplerate

    stream = sd.OutputStream(samplerate=fs, channels=2, callback=callback, blocksize=block_size)
    with stream:
        stream.start()
        print("Playback started... press Ctrl+C to stop.")
        input('Press Enter to stop playback...')

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
