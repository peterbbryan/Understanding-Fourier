import matplotlib
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn")  # switch to seaborn style


def _figure_1(output_path: str = "./figure_1.png"):
    # pylint: disable=too-many-locals
    """
    P

    Args:
    """

    noise_floor = 100
    x_offset, y_offset = 5, -20

    xs = np.arange(0, 2 * np.pi, 0.01)
    noiseless_sinusoid = 0.5 * np.cos(5 * xs)
    noised_sinusoid = noiseless_sinusoid + np.random.normal(
        loc=0, scale=0.5, size=noiseless_sinusoid.shape
    )

    fft_complex_noiseless = np.fft.fft(noiseless_sinusoid)
    fft_complex_noiseless_magnitude = np.abs(fft_complex_noiseless)[
        : len(fft_complex_noiseless) // 2
    ]
    fft_complex_noised = np.fft.fft(noised_sinusoid)
    fft_complex_noised_magnitude = np.abs(fft_complex_noised)[
        : len(fft_complex_noised) // 2
    ]

    fft_complex_denoised = fft_complex_noised.copy()
    fft_complex_denoised[np.abs(fft_complex_noised) < noise_floor] = 0
    denoised_sinusoid = np.fft.ifft(fft_complex_denoised).real

    plt.figure(figsize=(10, 8))

    plt.subplot(5, 1, 1)
    plt.plot(noiseless_sinusoid)
    plt.ylabel("Noiseless signal")
    plt.ylim(-1.5, 1.5)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(5, 1, 2)
    plt.plot(noised_sinusoid)
    plt.ylabel("Noised signal")
    plt.ylim(-1.5, 1.5)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(5, 1, 3)
    plt.plot(fft_complex_noiseless_magnitude)
    plt.ylabel("Noiseless FFT")
    plt.text(
        np.argmax(fft_complex_noiseless_magnitude) + x_offset,
        np.max(fft_complex_noiseless_magnitude) + y_offset,
        np.argmax(fft_complex_noiseless_magnitude),
    )
    plt.yticks([])

    plt.subplot(5, 1, 4)
    plt.plot(fft_complex_noised_magnitude)
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    plt.gca().add_patch(
        matplotlib.patches.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            noise_floor,
            linewidth=0,
            facecolor="red",
            alpha=0.1,
        )
    )
    plt.gca().add_patch(
        matplotlib.patches.Rectangle(
            (x_min, noise_floor),
            x_max - x_min,
            y_max - noise_floor,
            linewidth=0,
            facecolor="green",
            alpha=0.1,
        )
    )
    plt.ylabel("Noised FFT")
    plt.text(
        np.argmax(fft_complex_noised_magnitude) + x_offset,
        np.max(fft_complex_noised_magnitude) + y_offset,
        np.argmax(fft_complex_noised_magnitude),
    )
    plt.yticks([])

    plt.subplot(5, 1, 5)
    plt.plot(denoised_sinusoid)
    plt.ylabel("Denoised signal")
    plt.ylim(-1.5, 1.5)
    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()
    plt.savefig(output_path)


if __name__ == "__main__":
    _figure_1()
