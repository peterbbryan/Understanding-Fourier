"""
Basic realtime fft plotting.
"""

from typing import Generator

import matplotlib.pyplot as plt
import numpy as np
import pyaudio
from dynaconf import settings
from scipy.ndimage import gaussian_filter1d

plt.style.use("seaborn")  # switch to seaborn style


def _configure_mic(
    pyaudio_format: int = settings.FORMAT,
    n_channels: int = settings.N_CHANNELS,
    rate: int = settings.RATE,
    chunk_length: int = settings.CHUNK_LENGTH,
) -> pyaudio.Stream:
    """
    Configure your microphone to process each audio chunk.

    Args:
        pyaudio_format: Audio format of pyaudio microphone reader.
        n_channels: Number of input audio channels.
        rate: Sampling rate.
        chunk_length: Specifies the number of frames per buffer.
    """

    return pyaudio.PyAudio().open(
        format=pyaudio_format,
        channels=n_channels,
        rate=rate,
        input=True,
        output=True,
        frames_per_buffer=chunk_length,
    )


def _fourier_transform(data: np.ndarray, smoothing_coefficient: int = 3) -> np.ndarray:
    """
    Call FFT on data and add smoothing for visualization.

    Args:
        data: Data to run FFT on.
        smoothing_coefficient: Gaussian kernel smoothing width.
    Returns:
        np.ndarray of FFT returns.
    """

    return gaussian_filter1d(
        np.abs(np.fft.fft(np.frombuffer(data, np.int16))), smoothing_coefficient
    )


def _get_chunk_generator(
    stream: pyaudio.Stream, chunk_length: int = settings.CHUNK_LENGTH
) -> Generator[np.ndarray, None, None]:
    """
    Get generator for microphone audio data in increments.

    Args:
        stream: PyAudio stream to collect chunks from.
        chunk_length: Specifies the number of frames per buffer.
    Returns:
        Generator yielding microphone np.ndarray chunk.
    """

    while True:
        data = stream.read(chunk_length, exception_on_overflow=False)
        data = np.frombuffer(data, np.int16)

        yield data


def _plot_temporal_domain(
    ax: plt.Axes,
    temporal_data: np.ndarray,
    temporal_vlim: float = settings.TEMPORAL_VLIM,
) -> None:
    """
    Plot the raw waveform.

    Args:
        ax: Axis to plot to.
        temporal_data: Temporal data to plot.
        temporal_vlim: Min/max for plot.
    """

    ax.plot(temporal_data)
    ax.axhline(y=0.0, color="gray", linestyle="-")
    ax.set_xlim(0, len(temporal_data))
    ax.set_ylim(-temporal_vlim, temporal_vlim)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylabel("Signal waveform")


def _plot_frequency_domain(  # pylint: disable=too-many-arguments
    ax: plt.Axes,
    frequency_data: np.ndarray,
    waveform_vlim: float = settings.WAVEFORM_VLIM,
    rate: int = settings.RATE,
    chunk_length: int = settings.CHUNK_LENGTH,
    text_label_offset_x: int = 20,
) -> None:
    """
    Plot the magnitude of the positive frequency components of the discrete FFT.

    Args:
        ax: Axis to plot to.
        frequency_data: Frequency data to plot.
    """

    max_x = np.argmax(frequency_data)
    max_y = np.max(frequency_data)

    rescale_factor: float = rate / chunk_length

    ax.scatter(rescale_factor * max_x, max_y, color="red", marker="x")
    ax.text(
        rescale_factor * (max_x + text_label_offset_x),
        max_y,
        max_x * rescale_factor,
        color="red",
        backgroundcolor="white",
        alpha=0.8,
    )
    ax.set_xlim(0, rescale_factor * len(frequency_data) // 2)
    ax.set_ylim(1, waveform_vlim)
    ax.plot(rescale_factor * np.arange(len(frequency_data)), frequency_data)
    ax.set_yscale("log")
    ax.set_yticks([])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Log-scaled magnitude")


def plot_temporal_data_and_fft(wait_duration: float = 1e-10) -> None:
    """
    Plot the raw waveform and the FFT data.

    Args:
        wait_duration: Delay between plot updates.
    """

    stream: pyaudio.Stream = _configure_mic()
    chunk_generator = _get_chunk_generator(stream=stream)

    while True:

        plt.clf()

        data = next(chunk_generator)
        fft_data = _fourier_transform(data)

        ax1 = plt.subplot(2, 1, 1)
        _plot_temporal_domain(ax1, data)

        ax2 = plt.subplot(2, 1, 2)
        _plot_frequency_domain(ax2, fft_data)

        plt.pause(wait_duration)


if __name__ == "__main__":
    plot_temporal_data_and_fft()
