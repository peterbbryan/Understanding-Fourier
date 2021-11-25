import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn")  # switch to seaborn style


def _figure_1(output_path: str = "./figure_1.png"):
    """
    Plot waveform data.

    Args:
        output_path: Path to write figure to.
    """

    xs = np.arange(0, 2 * np.pi, 0.01)
    sinusoid_one = 0.5 * np.cos(5 * xs)
    sinusoid_two = 0.5 * np.cos(25 * xs)
    complex_tone = sinusoid_one + sinusoid_two

    plt.figure()

    plt.subplot(3, 1, 1)
    plt.plot(sinusoid_one)
    plt.ylabel("Pure tone 1")
    plt.ylim(-1.5, 1.5)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(3, 1, 2)
    plt.plot(sinusoid_two)
    plt.ylabel("Pure tone 2")
    plt.ylim(-1.5, 1.5)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(3, 1, 3)
    plt.plot(complex_tone)
    plt.ylabel("Complex tone")
    plt.ylim(-1.5, 1.5)
    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()
    plt.savefig(output_path)


def _figure_2(output_path: str = "./figure_2.png"):
    """
    Plot FFT data.

    Args:
        output_path: Path to write figure to.
    """

    x_offset, y_offset = 5, -10

    xs = np.arange(0, 2 * np.pi, 0.01)
    sinusoid_one = 0.5 * np.cos(5 * xs)
    fft_one = np.fft.fft(sinusoid_one)
    fft_one_magnitude = np.abs(fft_one)[: len(fft_one) // 2]
    
    sinusoid_two = 0.5 * np.cos(25 * xs)
    fft_two = np.fft.fft(sinusoid_two)
    fft_two_magnitude = np.abs(fft_two)[: len(fft_two) // 2]
    
    complex_tone = sinusoid_one + sinusoid_two
    fft_complex = np.fft.fft(complex_tone)
    fft_complex_magnitude = np.abs(fft_complex)[: len(fft_complex) // 2]

    plt.figure()

    plt.subplot(3, 1, 1)
    plt.plot(fft_one_magnitude)
    plt.ylabel("Pure tone 1")
    plt.text(
        np.argmax(fft_one_magnitude) + x_offset,
        np.max(fft_one_magnitude) + y_offset,
        np.argmax(fft_one_magnitude),
    )
    plt.yticks([])

    plt.subplot(3, 1, 2)
    plt.plot(fft_two_magnitude)
    plt.ylabel("Pure tone 2")
    plt.text(
        np.argmax(fft_two_magnitude) + x_offset,
        np.max(fft_two_magnitude) + y_offset,
        np.argmax(fft_two_magnitude),
    )
    plt.yticks([])

    plt.subplot(3, 1, 3)
    plt.plot(fft_complex_magnitude)
    plt.ylabel("Complex tone")
    plt.yticks([])

    plt.tight_layout()
    plt.savefig(output_path)


if __name__ == "__main__":
    _figure_1()
    _figure_2()
