import numpy as np


def abs(complex_number: complex) -> float:  # pylint: disable=redefined-builtin
    """
    Get the magnitude (abs) of complex number.

    Args:
        complex_number: Complex number.
    Returns:
        Magnitude.
    """

    real_component = complex_number.real
    imaginary_component = complex_number.imag

    # calculate distance to origin
    sum_of_squares = (real_component ** 2) + (imaginary_component ** 2)
    sqrt_of_sum_of_squares = sum_of_squares ** (1 / 2)

    return sqrt_of_sum_of_squares


assert abs(1 + 1j) == np.abs(1 + 1j)
assert abs(2 * (1 - 1j)) == np.abs(2 * (1 - 1j))
assert abs(3 * (-1 + 1j)) == np.abs(3 * (-1 + 1j))
assert abs(4 * (-1 - 1j)) == np.abs(4 * (-1 - 1j))


def angle(complex_number: complex) -> float:
    """
    Get phase (angle) of complex number.

    Args:
        complex_number: Complex number.
    Returns:
        Phase angle.
    """

    real_component = complex_number.real
    imaginary_component = complex_number.imag

    return np.arctan2(imaginary_component, real_component)


assert angle(1 + 1j) == np.angle(1 + 1j)
assert angle(2 * (1 - 1j)) == np.angle(2 * (1 - 1j))
assert angle(3 * (-1 + 1j)) == np.angle(3 * (-1 + 1j))
assert angle(4 * (-1 - 1j)) == np.angle(4 * (-1 - 1j))
