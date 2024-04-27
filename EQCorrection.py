import numpy as np
import scipy.signal as signal
import pyaudio
import wave
import time

def process_block(block, fs, filters):
    """Apply filters to a block of audio and measure processing time."""
    start_time = time.time()
    output = np.zeros_like(block, dtype=np.float32)  # Ensure output array is float for safe accumulation
    for b in filters:
        filtered_data = signal.lfilter(b, [1.0], block)
        output += filtered_data  # Already in float32, safe to add directly
    processing_time = time.time() - start_time
    return output, processing_time

def main(filename, num_filters):
    wf = wave.open(filename, 'rb')

    # Print file details
    print(f"File: {filename}")
    print(f"Channels: {wf.getnchannels()}")
    print(f"Sample Width: {wf.getsampwidth()} bytes")
    print(f"Sample Rate: {wf.getframerate()} Hz")
    print(f"Number of Frames: {wf.getnframes()}")
    print(f"Compression Type: {wf.getcomptype()}")
    print(f"Compression Name: {wf.getcompname()}")

    # Setup PyAudio stream
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # Filter setup
    numtaps = 101  # Filter length, can be adjusted
    low_freq = 20
    high_freq = 20000
    fs = wf.getframerate()
    frequencies = np.logspace(np.log10(low_freq), np.log10(high_freq), num_filters+1)
    filters = [signal.firwin(numtaps, [frequencies[i], frequencies[i+1]], fs=fs, pass_zero=False) for i in range(num_filters)]

    blocksize = 1024  # Define block size for processing
    data = wf.readframes(blocksize)

    while len(data) > 0:
        numpydata = np.frombuffer(data, dtype=np.int16)  # Correct dtype for 16-bit data
        numpydata = numpydata.astype(np.float32) / 32768  # Normalize to -1.0 to 1.0 range
        processed_data, proc_time = process_block(numpydata, fs, filters)
        block_duration = blocksize / fs
        proc_ratio = proc_time / block_duration
        print(f"Processing Time: {proc_time:.6f} seconds, Processing Ratio: {proc_ratio:.2f}")
        
        if proc_ratio > 1.0:
            print(f"WARNING: Processing time ratio {proc_ratio:.2f} exceeds real-time capability.")
        elif proc_ratio > 0.75:
            print(f"NOTICE: High load, processing time ratio is {proc_ratio:.2f}. Close to capacity.")
        
        stream.write(processed_data.tobytes())
        data = wf.readframes(blocksize)

    stream.stop_stream()
    stream.close()
    p.terminate()
    wf.close()

filename = 'wavfiles\whitenoise.wav'  # Path to your WAV file
num_filters = 10  # Number of bandpass filters
main(filename, num_filters)
