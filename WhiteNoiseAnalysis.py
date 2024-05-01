import pyaudio
import wave
import threading
import time
import numpy as np
import matplotlib.pyplot as plt

# Settings
FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1             # Number of audio channels
RATE = 44100              # Sample rate
CHUNK = 1024              # Number of audio frames per buffer
PLAY_SECONDS = 3          # Length of time in seconds to play whitenoise.wav
RECORD_SECONDS = 3        # Length of time in seconds to record, can be different from PLAY_SECONDS
WAVE_OUTPUT_FILENAME = "wavfiles/measurement.wav"
WAVE_INPUT_FILENAME = "wavfiles/whitenoise.wav"

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

# Record audio function with added delay
def record_audio(record_time):
    # Delay to allow playback to start
    time.sleep(0.5)  # Adjust this delay as needed
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

# FFT analysis function
def fft_analysis(data, fs, n_fft=8192):
    """Compute the FFT of the data and return frequency bins and magnitude spectrum."""
    fft_data = np.fft.rfft(data, n_fft)
    freq = np.fft.rfftfreq(n_fft, 1/fs)
    magnitude = np.abs(fft_data)
    return freq, magnitude

# Plot FFT of the recorded audio from WAV file
def plot_fft_from_file(filename):
    wf = wave.open(filename, 'rb')
    data = wf.readframes(wf.getnframes())
    audio_data = np.frombuffer(data, dtype=np.int16)
    if wf.getnchannels() == 2:
        audio_data = audio_data[0::2]
    freq, magnitude = fft_analysis(audio_data, RATE)
    plt.figure(figsize=(10, 5))
    plt.plot(freq, magnitude)
    plt.title('FFT of Recorded Audio')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.xlim(0, RATE / 2)  # Limit x-axis to Nyquist frequency
    plt.show()
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
    
    # After threads are done, plot FFT from WAV file
    plot_fft_from_file(WAVE_OUTPUT_FILENAME)
