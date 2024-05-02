import os
import soundfile as sf
import sounddevice as sd

def play_audio(file_path):
    # Read the audio file
    data, fs = sf.read(file_path)
    
    # Play audio
    sd.play(data, fs)
    sd.wait()  # Wait until file is done playing

def main():
    directory = 'songs'  # The directory containing the songs
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
