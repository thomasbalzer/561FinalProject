import os
import numpy as np
import pandas as pd
import sounddevice as sd
import soundfile as sf
from scipy.signal import lfilter

def load_filters(csv_file):
    df = pd.read_csv(csv_file)
    filters = []
    gains = []
    for _, row in df.iterrows():
        gain = row['Adjustment Factor']
        coefficients = np.array(row[2:], dtype=float)
        filters.append(coefficients)
        gains.append(gain)
    return filters, gains

def apply_filters(data, filters, gains):
    filtered_data = np.zeros_like(data)
    for b, gain in zip(filters, gains):
        filtered_output = lfilter(b, [1.0], data)
        filtered_data += filtered_output * gain
    # Normalize to prevent clipping and ensure audible output
    max_amp = np.max(np.abs(filtered_data))
    if max_amp > 0:
        filtered_data /= max_amp
    return filtered_data

def play_audio(file_path, filters, gains, buffer_size=1024):  # Adjust buffer size if needed
    data, fs = sf.read(file_path, dtype='float32')
    if data.ndim > 1:
        data = data.mean(axis=1)  # Convert to mono if stereo
    print(f"Sample rate: {fs}")

    def callback(outdata, frames, time, status):
        nonlocal data
        if status:
            print(f"Status: {status}")
        if len(data) == 0:
            outdata.fill(0)
            raise sd.CallbackStop
        valid_frames = min(len(data), frames)
        outdata[:valid_frames] = apply_filters(data[:valid_frames], filters, gains).reshape(-1, 1)
        outdata[valid_frames:] = 0  # Zero-fill the rest of the buffer if needed
        data = data[valid_frames:]  # Move forward in the data

    with sd.OutputStream(samplerate=fs, channels=1, callback=callback, blocksize=buffer_size) as stream:
        print("Playback started... press Ctrl+C to stop.")
        while stream.active:
            sd.sleep(100)  # Keep this thread alive while the audio is playing

def find_and_play_all_wav_files(directory, filters, gains):
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):
            filepath = os.path.join(directory, filename)
            print(f"Playing: {filepath}")
            play_audio(filepath, filters, gains)

def main():
    directory = 'songs'
    csv_file = 'filter_coefficients.csv'
    filters, gains = load_filters(csv_file)
    find_and_play_all_wav_files(directory, filters, gains)

if __name__ == "__main__":
    main()
