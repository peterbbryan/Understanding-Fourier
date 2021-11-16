"""
Basic realtime fft plotting.
"""

import matplotlib.pyplot as plt
import numpy as np
import pyaudio
from scipy.ndimage import gaussian_filter1d

plt.style.use("seaborn")  # switch to seaborn style

CHUNK = 1000
_RUNNING_MAX = 1e6
_RUNNING_MAX_VOLUME = 35000
_FORMAT = pyaudio.paInt16
_CHANNELS = 1
_RATE = 10000


def _configure_mic() -> pyaudio.Stream:
    """
    """

    mic = pyaudio.PyAudio()

    return mic.open(
        format=_FORMAT,
        channels=_CHANNELS,
        rate=_RATE,
        input=True,
        output=True,
        frames_per_buffer=CHUNK,
    )


def _get_chunk_generator():
    ...


def _plot_temporal_domain(ax: plt.Axes, temporal_data: np.ndarray):
    """

    Args:
        ax:
        data:
    """

    ax.plot(temporal_data)
    ax.axhline(y=0.0, color="gray", linestyle="-")
    ax.set_xlim(0, CHUNK)
    ax.set_ylim(-_RUNNING_MAX_VOLUME, _RUNNING_MAX_VOLUME)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel("")


def _plot_frequency_domain(ax: plt.Axes, frequency_data: np.ndarray):
    """
    """

    ax.scatter(max_x * 10, max_y, color="red", marker="x")
    ax.text(
        10 * (max_x + 20),
        max_y,
        max_x * 10,
        color="red",
        backgroundcolor="white",
        alpha=0.8,
    )
    ax.set_xlim(0, 10 * len(frequency_data) // 2)
    ax.set_ylim(1, _RUNNING_MAX)
    ax.plot(np.arange(len(frequency_data)) * 10, frequency_data)
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

    data = stream.read(CHUNK, exception_on_overflow=False)
    data = np.frombuffer(data, np.int16)
    fft_data = gaussian_filter1d(np.abs(np.fft.fft(np.frombuffer(data, np.int16))), 5)

    max_x = np.argmax(fft_data)
    max_y = np.max(fft_data)
    plt.clf()
    ax1 = plt.subplot(2, 1, 1)
    _plot_temporal_domain(ax1, data)
    ax2 = plt.subplot(2, 1, 2)
    _plot_frequency_domain(ax2, fft_data)
    plt.pause(0.0000001)
