
import dynaconf
import pyaudio
import struct
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scipy.ndimage import gaussian_filter1d

plt.style.use('seaborn')     # switch to seaborn style

CHUNK = 1000


def _configure_mic() -> pyaudio.Stream:
    """
    """
    
    mic = pyaudio.PyAudio()
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 10000
    stream = mic.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, output=True, frames_per_buffer=CHUNK)

    return stream


def _get_chunk_generator():
    ...


def _plot_temporal_domain(ax: plt.Axes, temporal_data: np.ndarray):
    """
    
    Args:
        ax:
        data:
    """
    
    ax.plot(data)
    ax.axhline(y=0.0, color='gray', linestyle='-')
    ax.set_xlim(0, CHUNK)
    ax.set_ylim(-running_max_volume, running_max_volume)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel("")


def _plot_frequency_domain(ax: plt.Axes, frequency_data: np.ndarray):
    """
    """
    
    ax.scatter(max_x*10, max_y, color="red", marker="x")
    ax.text(10*(max_x+20), max_y, max_x*10, color="red", backgroundcolor="white", alpha=0.8)
    ax.set_xlim(0, 10*len(fft_data)//2)
    ax.set_ylim(1, running_max)
    ax.plot(np.arange(len(fft_data))*10, fft_data)
    ax.set_yscale("log")
    ax.set_yticks([])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Log-scaled magnitude")

def _fourier_transform():
    """
    """
    
    ...


stream: pyaudio.Stream = _configure_mic()

while True:

    running_max = 1e6
    running_max_volume = 35000

    data = stream.read(CHUNK, exception_on_overflow = False)
    data = np.frombuffer(data, np.int16)
    temporal_data = np.frombuffer(data, np.int16)
    fft_data = gaussian_filter1d(np.abs(np.fft.fft(temporal_data)), 5)

    running_max = running_max if running_max > np.max(fft_data) else np.max(fft_data)
    running_max_volume = running_max_volume if running_max_volume > np.max(data) else np.max(data)

    max_x = np.argmax(fft_data)
    max_y = np.max(fft_data)
    plt.clf()
    ax = plt.subplot(2,1,1)
    _plot_temporal_domain(ax, data)
    ax2 = plt.subplot(2,1,2)
    _plot_frequency_domain(ax2, fft_data)
    plt.pause(0.0000001)

