import pyaudio
import wave
import numpy as np
import matplotlib.pyplot as plt
import threading
import time

# Function to play WAV file with adjustable length
def play_wav(filename, play_seconds, volume=0.1):
    wf = wave.open(filename, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    frames_per_buffer = 1024
    total_frames = int(wf.getframerate() / frames_per_buffer * play_seconds)
    played_frames = 0

    data = wf.readframes(frames_per_buffer)
    while data and played_frames < total_frames:
        audio_data = np.frombuffer(data, dtype=np.int16)  # Assumes 16-bit audio
        lowered_audio_data = (audio_data * volume).astype(np.int16)
        stream.write(lowered_audio_data.tobytes())
        data = wf.readframes(frames_per_buffer)
        played_frames += 1

    stream.stop_stream()
    stream.close()
    p.terminate()

# Function to record audio and perform FFT analysis with adjustable recording length
def record_audio_and_fft(output_filename, record_seconds):
    FORMAT = pyaudio.paInt24  # 24-bit format
    CHANNELS = 1
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
    raw_bytes = b''.join(frames)
    frames_numpy = np.frombuffer(raw_bytes, dtype=np.int24).view(np.float32)  # Proper type for 24-bit
    fft_result = np.fft.rfft(frames_numpy)
    freqs = np.fft.rfftfreq(len(frames_numpy), 1/RATE)

    plt.figure()
    plt.plot(freqs, np.abs(fft_result))
    plt.title("Frequency Spectrum of the Recorded Audio")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.xscale('linear')
    plt.yscale('log')
    plt.xlim(20, 20000)  # Audible range
    plt.show()

# User input for durations
play_seconds = 2
record_seconds = 2

filename_to_play = 'wavfiles/whitenoise.wav'
output_filename = 'wavfiles/measurement.wav'

# Start recording in a separate thread
recording_thread = threading.Thread(target=record_audio_and_fft, args=(output_filename, record_seconds))
recording_thread.start()

# Wait briefly for the recording to start
time.sleep(0.5)

# Play the WAV file
play_wav(filename_to_play, play_seconds)

# Wait for recording to finish
recording_thread.join()

print("Playback and recording finished.")
