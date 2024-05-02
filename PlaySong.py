import soundfile as sf
import sounddevice as sd

def play_audio(file_path):
    # Read the audio file
    data, fs = sf.read(file_path)
    
    # Play audio
    sd.play(data, fs)
    sd.wait()  # Wait until file is done playing

def main():
    file_path = 'songs\Kid Charlemagne.wav'  # Change this to the path of your song
    print(f"Playing: {file_path}")
    play_audio(file_path)

if __name__ == "__main__":
    main()
