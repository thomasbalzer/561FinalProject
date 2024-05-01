import pyaudio
import wave
import numpy as np
import matplotlib.pyplot as plt
import threading
import time

# Function to play WAV file
def play_wav(filename):
    wf = wave.open(filename, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    data = wf.readframes(1024)
    while data:
        stream.write(data)
        data = wf.readframes(1024)
    stream.stop_stream()
    stream.close()
    p.terminate()

# Modified function to record audio with a different format and perform FFT analysis
def record_audio_and_fft(output_filename, record_seconds):
    FORMAT = pyaudio.paInt24  # Change to 24-bit format
    CHANNELS = 2
    RATE = 44100
    CHUNK = 1024
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("Recording...")
    frames = []
    for _ in range(0, int(RATE / CHUNK * record_seconds)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("Finished recording.")
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(output_filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    # FFT analysis
    raw_bytes = b''.join(frames)  # Concatenate all frame bytes
    # Assemble bytes into an array of 32-bit integers
    frames_numpy = np.frombuffer(raw_bytes, dtype=np.uint8).view(np.int32)
    fft_result = np.fft.rfft(frames_numpy)
    freqs = np.fft.rfftfreq(len(frames_numpy), 1/RATE)
    
    plt.figure()
    plt.plot(freqs, np.abs(fft_result))
    plt.title("Frequency spectrum of the recorded audio")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.xlim(20, 20000)
    plt.xscale('log')
    plt.show()

# Play the WAV file and record simultaneously
filename_to_play = 'wavfiles/whitenoise.wav'
output_filename = 'wavefiles/measurement.wav'
record_seconds = 3  # Set to 3 seconds

# Start recording in a separate thread
recording_thread = threading.Thread(target=record_audio_and_fft, args=(output_filename, record_seconds))
recording_thread.start()

# Wait briefly for the recording to start
time.sleep(0.5)

# Play the WAV file
play_wav(filename_to_play)

# Wait for recording to finish
recording_thread.join()

print("Playback and recording finished.")
