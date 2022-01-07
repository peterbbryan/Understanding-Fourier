import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn")  # switch to seaborn style


def _figure_1(output_path: str = "./figure_1.png") -> None:
    """
    Plot phase and magnitude.

    Args:
        output_path: Path to write figure to.
    """

    plt.figure()

    # plot a complex number sample
    plt.scatter(1, 1, s=20)
    plt.plot([(0, 0), (1, 1)], linewidth=0.5, color="blue")
    plt.text(1.05, 1.05, "1+1j")

    # plot angle
    theta = np.linspace(0, np.pi / 4, 10)
    x1 = 0.5 * np.cos(theta)
    x2 = 0.5 * np.sin(theta)
    plt.plot(x1, x2, linewidth=0.3, color="gray")
    plt.text(0.5, 0.2, f"arctan2(imag, real) = \n{np.arctan2(1, 1):.2f}")

    # plot distance
    plt.text(
        0.06, 0.16, f"sqrt(imag^2 + real^2) = {np.sqrt(1**2 + 1**2):.2f}", rotation=45
    )

    # rectify the axes
    plt.axhline(y=0, color="gray", linestyle="-")
    plt.axvline(x=0, color="gray", linestyle="-")
    plt.axis("square")
    plt.xlabel("Real component")
    plt.xlim(-1.4, 1.4)
    plt.ylabel("Imaginary component")
    plt.ylim(-1.4, 1.4)

    plt.tight_layout()
    plt.savefig(output_path)


if __name__ == "__main__":
    _figure_1()
