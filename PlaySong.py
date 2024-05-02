import os
import numpy as np
import soundfile as sf
import sounddevice as sd

def audio_stream_generator(file_path, block_size):
    with sf.SoundFile(file_path) as sf_file:
        while True:
            data = sf_file.read(block_size)
            if len(data) == 0:
                break
            yield data

def play_audio(file_path):
    block_size = 1024  # Block size for chunks
    data_generator = audio_stream_generator(file_path, block_size)

    def callback(outdata, frames, time, status):
        try:
            data = next(data_generator)
            if len(data) < outdata.shape[0]:
                # If the last block is shorter than the block size, pad with zeros
                data = np.pad(data, (0, outdata.shape[0] - len(data)), mode='constant')
            outdata[:] = data
        except StopIteration:
            outdata.fill(0)  # Fill remaining buffer with zeros
            raise sd.CallbackStop

    # Get the sample rate and channel count from the file
    with sf.SoundFile(file_path) as sf_file:
        fs = sf_file.samplerate
        channels = sf_file.channels

    # Setting up the audio stream for playback
    with sd.OutputStream(samplerate=fs, channels=channels, callback=callback, blocksize=block_size):
        input("Press Enter to stop playback...")  # Keeps the stream active until Enter is pressed

def main():
    directory = 'songs'  # Directory containing the songs
    songs = [f for f in os.listdir(directory) if f.endswith('.wav')]
    
    if not songs:
        print("No songs found in the directory.")
        return

    # Just playing the first song for demonstration
    song_path = os.path.join(directory, songs[0])
    print(f"Playing: {song_path}")
    play_audio(song_path)

if __name__ == "__main__":
    main()
