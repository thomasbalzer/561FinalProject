import pyaudio
import wave
import threading
import time

# Settings
FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1              # Number of audio channels
RATE = 44100              # Bit Rate
CHUNK = 1024              # Number of audio frames per buffer
PLAY_SECONDS = 3          # Length of time in seconds to play whitenoise.wav
RECORD_SECONDS = 3        # Length of time in seconds to record, can be different from PLAY_SECONDS
WAVE_OUTPUT_FILENAME = "wavfiles\measurement.wav"
WAVE_INPUT_FILENAME = "wavfiles\whitenoise.wav"

# Initialize pyaudio
audio = pyaudio.PyAudio()

# Play audio function
def play_audio(play_time):
    wf = wave.open(WAVE_INPUT_FILENAME, 'rb')
    stream = audio.open(format=audio.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)
    start_time = time.time()
    data = wf.readframes(CHUNK)
    while data != b'' and time.time() - start_time < play_time:
        stream.write(data)
        data = wf.readframes(CHUNK)
    stream.stop_stream()
    stream.close()

# Record audio function
def record_audio(record_time):
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    frames = []
    for i in range(0, int(RATE / CHUNK * record_time)):
        data = stream.read(CHUNK)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

# Start play and record threads
if __name__ == "__main__":
    t1 = threading.Thread(target=play_audio, args=(PLAY_SECONDS,))
    t2 = threading.Thread(target=record_audio, args=(RECORD_SECONDS,))

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    audio.terminate()
