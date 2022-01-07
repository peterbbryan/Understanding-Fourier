import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from skimage import data

plt.style.use("seaborn")  # switch to seaborn style


def _figure_1(output_path: str = "./figure_1.png") -> None:
    # pylint: disable=too-many-locals
    """
    Plot FFT noise suppression below some power level.

    Args:
        output_path: Path to write figure to.
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
            noise_floor - y_min,
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


def _figure_2(output_path: str = "./figure_2.png") -> None:
    # pylint: disable=too-many-locals,too-many-statements
    """
    Plot FFT noise suppression with a bandpass filter.

    Args:
        output_path: Path to write figure to.
    """

    minimum_frequency = 1
    maximum_frequency = 10
    x_offset, y_offset = 5, -20

    xs = np.arange(0, 2 * np.pi, 0.01)
    noiseless_sinusoid = 0.5 * np.cos(5 * xs)
    noised_sinusoid = noiseless_sinusoid.copy()
    noised_sinusoid += 1 * np.cos(100 * xs)
    noised_sinusoid += 1 * np.cos(130 * xs)
    noised_sinusoid += 1 * np.cos(200 * xs)

    fft_complex_noiseless = np.fft.fft(noiseless_sinusoid)
    fft_complex_noiseless_magnitude = np.abs(fft_complex_noiseless)[
        : len(fft_complex_noiseless) // 2
    ]
    fft_complex_noised = np.fft.fft(noised_sinusoid)
    fft_complex_noised_magnitude = np.abs(fft_complex_noised)[
        : len(fft_complex_noised) // 2
    ]

    # this would be simplified if we used rfft, but that is in the next article!
    # since we didn't use it, we are going to have to account for the negative components
    # https://docs.scipy.org/doc/scipy/reference/tutorial/fft.html
    fft_complex_denoised = fft_complex_noised.copy()
    fft_complex_denoised[0:minimum_frequency] = 0
    fft_complex_denoised[maximum_frequency : len(fft_complex_denoised) // 2] = 0
    fft_complex_denoised[len(fft_complex_denoised) // 2 : -maximum_frequency] = 0
    fft_complex_denoised[-minimum_frequency:] = 0
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
    plt.ylim(-3, 3)
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
            minimum_frequency - x_min,
            y_max - y_min,
            linewidth=0,
            facecolor="red",
            alpha=0.1,
        )
    )
    plt.gca().add_patch(
        matplotlib.patches.Rectangle(
            (minimum_frequency, y_min),
            maximum_frequency - minimum_frequency,
            y_max - y_min,
            linewidth=0,
            facecolor="green",
            alpha=0.1,
        )
    )
    plt.gca().add_patch(
        matplotlib.patches.Rectangle(
            (maximum_frequency, y_min),
            x_max - maximum_frequency,
            y_max - y_min,
            linewidth=0,
            facecolor="red",
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


def _figure_3(output_path: str = "./figure_3.png") -> None:
    # pylint: disable=too-many-locals
    """
    Plot high frequency content in image.

    Args:
        output_path: Path to write figure to.
    """

    im_data = data.camera()
    im_fft = np.fft.fft2(im_data)
    fft_min = np.log(np.abs(im_fft).min())
    fft_max = np.log(np.abs(im_fft).max())
    im_fft_shifted = np.fft.fftshift(im_fft)

    x, y = np.meshgrid(np.arange(im_data.shape[1]), np.arange(im_data.shape[0]))
    radii = np.sqrt((x - im_data.shape[1] // 2) ** 2 + (y - im_data.shape[1] // 2) ** 2)

    im_fft_reconstructed_shifted = im_fft_shifted.copy()
    im_fft_reconstructed_shifted[radii < 70] = np.finfo(float).eps
    im_fft_reconstructed = np.fft.ifftshift(im_fft_reconstructed_shifted)
    im_data_reconstructed = np.abs(np.fft.ifft2(im_fft_reconstructed))

    plt.figure(figsize=(10, 3))

    plt.subplot(1, 4, 1)
    plt.title("Original image")
    plt.imshow(im_data, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.title("Log-scaled FFT magnitude")
    plt.imshow(np.log(np.abs(im_fft_shifted)), cmap="gray", vmin=fft_min, vmax=fft_max)
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.title("Low freqs removed")
    plt.imshow(
        np.log(np.abs(im_fft_reconstructed_shifted)),
        cmap="gray",
        vmin=fft_min,
        vmax=fft_max,
    )
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.title("Visualized high frequencies")
    plt.imshow(im_data_reconstructed, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path)


def _figure_4(output_path: str = "./figure_4.png") -> None:
    # pylint: disable=too-many-locals
    """
    Plot low frequency content in image.

    Args:
        output_path: Path to write figure to.
    """

    im_data = data.camera()
    im_fft = np.fft.fft2(im_data)
    fft_min = np.log(np.abs(im_fft).min())
    fft_max = np.log(np.abs(im_fft).max())
    im_fft_shifted = np.fft.fftshift(im_fft)

    x, y = np.meshgrid(np.arange(im_data.shape[1]), np.arange(im_data.shape[0]))
    radii = np.sqrt((x - im_data.shape[1] // 2) ** 2 + (y - im_data.shape[1] // 2) ** 2)

    im_fft_reconstructed_shifted = im_fft_shifted.copy()
    im_fft_reconstructed_shifted[radii > 10] = np.finfo(float).eps
    im_fft_reconstructed = np.fft.ifftshift(im_fft_reconstructed_shifted)
    im_data_reconstructed = np.abs(np.fft.ifft2(im_fft_reconstructed))

    plt.figure(figsize=(10, 3))

    plt.subplot(1, 4, 1)
    plt.title("Original image")
    plt.imshow(im_data, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.title("Log-scaled FFT magnitude")
    plt.imshow(np.log(np.abs(im_fft_shifted)), cmap="gray", vmin=fft_min, vmax=fft_max)
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.title("High freqs removed")
    plt.imshow(
        np.log(np.abs(im_fft_reconstructed_shifted)),
        cmap="gray",
        vmin=fft_min,
        vmax=fft_max,
    )
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.title("Visualized low frequencies")
    plt.imshow(im_data_reconstructed, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path)


if __name__ == "__main__":
    _figure_1()
    _figure_2()
    _figure_3()
    _figure_4()
