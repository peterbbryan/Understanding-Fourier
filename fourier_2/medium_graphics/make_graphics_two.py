import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn")  # switch to seaborn style


def _figure_1(output_path: str = "./figure_1.png") -> None:
    """
    Plot waveform data with phase estimates.

    Args:
        output_path: Path to write figure to.
    """

    xs = np.arange(0, 2 * np.pi, 0.01)

    sinusoid_one = 0.5 * np.cos(5 * xs)
    sinusoid_two = 0.5 * np.cos(1 + 5 * xs)
    sinusoid_three = 0.5 * np.cos(2 + 5 * xs)

    plt.figure()

    plt.subplot(3, 1, 1)
    plt.plot(sinusoid_one)
    plt.ylabel("0.5 * cos(5x)")
    plt.ylim(-1.5, 1.5)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(3, 1, 2)
    plt.plot(sinusoid_two)
    plt.ylabel("0.5 * cos(5x + 1)")
    plt.ylim(-1.5, 1.5)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(3, 1, 3)
    plt.plot(sinusoid_three)
    plt.ylabel("0.5 * cos(5x + 2)")
    plt.ylim(-1.5, 1.5)
    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()
    plt.savefig(output_path)


def _figure_2(output_path: str = "./figure_2.png") -> None:
    """
    Plot waveform data with phase estimates.

    Args:
        output_path: Path to write figure to.
    """

    xs = np.arange(0, 2 * np.pi, 0.01)

    sinusoid_one = 0.5 * np.cos(5 * xs)
    sinusoid_one_fft = np.fft.fft(sinusoid_one)
    sinusoid_one_fft_magnitude = np.abs(sinusoid_one_fft)
    sinusoid_one_fft_angle = np.angle(sinusoid_one_fft)

    sinusoid_two = 0.5 * np.cos(1 + 5 * xs)
    sinusoid_two_fft = np.fft.fft(sinusoid_two)
    sinusoid_two_fft_magnitude = np.abs(sinusoid_two_fft)
    sinusoid_two_fft_angle = np.angle(sinusoid_two_fft)

    sinusoid_three = 0.5 * np.cos(2 + 5 * xs)
    sinusoid_three_fft = np.fft.fft(sinusoid_three)
    sinusoid_three_fft_magnitude = np.abs(sinusoid_three_fft)
    sinusoid_three_fft_angle = np.angle(sinusoid_three_fft)

    plt.figure()

    plt.subplot(3, 1, 1)
    plt.plot(sinusoid_one)
    plt.ylabel("0.5 * cos(5x)")
    plt.title(
        "sinusoid_one_fft_magnitude = np.abs(sinusoid_one_fft) \n"
        + "sinusoid_one_fft_angle = np.angle(sinusoid_one_fft) \n"
        + "sinusoid_one_fft_angle[np.argmax(sinusoid_one_fft_magnitude)]: "
        + f"{sinusoid_one_fft_angle[np.argmax(sinusoid_one_fft_magnitude)]:.1f}"
    )
    plt.ylim(-1.5, 1.5)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(3, 1, 2)
    plt.plot(sinusoid_two)
    plt.ylabel("0.5 * cos(5x + 1)")
    plt.title(
        "sinusoid_two_fft_magnitude = np.abs(sinusoid_two_fft) \n"
        + "sinusoid_two_fft_angle = np.angle(sinusoid_two_fft) \n"
        + "sinusoid_two_fft_angle[np.argmax(sinusoid_two_fft_magnitude)]: "
        + f"{sinusoid_two_fft_angle[np.argmax(sinusoid_two_fft_magnitude)]:.1f}"
    )
    plt.ylim(-1.5, 1.5)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(3, 1, 3)
    plt.plot(sinusoid_three)
    plt.ylabel("0.5 * cos(5x + 2)")
    plt.title(
        "sinusoid_three_fft_magnitude = np.abs(sinusoid_three_fft) \n"
        + "sinusoid_three_fft_angle = np.angle(sinusoid_three_fft) \n"
        + "sinusoid_three_fft_angle[np.argmax(sinusoid_three_fft_magnitude)]: "
        + f"{sinusoid_three_fft_angle[np.argmax(sinusoid_three_fft_magnitude)]:.1f}"
    )
    plt.ylim(-1.5, 1.5)
    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()
    plt.savefig(output_path)


if __name__ == "__main__":
    _figure_1()
    _figure_2()
