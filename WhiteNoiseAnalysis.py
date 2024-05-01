import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import scipy.fftpack

# Constants
FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
CHANNELS = 1  # Single channel for microphone
RATE = 44100  # Sampling rate
CHUNK = 1024  # Number of audio samples per frame
DURATION = 5  # Duration in seconds to record

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Recording...")

frames = []

# Record for a few seconds
for _ in range(0, int(RATE / CHUNK * DURATION)):
    data = stream.read(CHUNK)
    frames.append(np.fromstring(data, dtype=np.int16))

print("Finished recording.")

# Stop and close the stream
stream.stop_stream()
stream.close()
p.terminate()

# Convert the list of numpy-arrays into a single numpy-array.
signal = np.concatenate(frames)

# FFT
N = len(signal)
yf = scipy.fftpack.fft(signal)
xf = np.linspace(0.0, 1.0/(2.0*(1/RATE)), N//2)

# Plot
plt.plot(xf, 2.0/N * np.abs(yf[:N//2]))
plt.grid()
plt.title("FFT of Recorded Signal")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.show()
