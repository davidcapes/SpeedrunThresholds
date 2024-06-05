import numpy as np
from numba import njit


@njit
def calculate_from_table(x, y_table, x_table_lwr, x_table_upr, rounding=8):
    """
    :param x: A floating point number.
    :param y_table: A numpy array of y values calculated for a set of equally spaced x values.
    :param x_table_lwr: The x value that corresponds to first entry in y_table.
    :param x_table_upr: The x value that corresponds to the last entry in y_table.
    :param rounding: How many decimal places to round the final result to. Used to eliminate floating point errors.

    :return: An estimate of y at the inputted x using the provided y_table and linear interpolation.
    """

    # Ensure x can be calculated from table.
    if not x_table_lwr <= x <= x_table_upr:
        print(f"CustomError: In calculate_from_table; {x} not in the bounds [", x_table_lwr, ",", x_table_upr, "]")
        return
    elif len(y_table) == 0:
        print(f"CustomError: In calculate_from_table; y_table is empty.")
        return
    elif x_table_lwr == x_table_upr:
        print(f"CustomError: In calculate_from_table; {str(x_table_lwr)}=x_table_lwr=x_table_upr={str(x_table_upr)}")
        return

    # Calculate indexes and bounds for x.
    index_estimate = (len(y_table) - 1) * (x - x_table_lwr) / (x_table_upr - x_table_lwr)
    index_lwr, index_upr = int(np.floor(index_estimate)), int(np.ceil(index_estimate))
    y_lwr, y_upr = y_table[index_lwr], y_table[index_upr]

    # Return calculation.
    if index_lwr == index_upr:
        return np.round(y_lwr, rounding)
    r = (index_estimate - index_lwr) / (index_upr - index_lwr)
    return np.round((1 - r) * y_lwr + r * y_upr, rounding)


def integral(f, a, b, n=1000):
    """
    :param f: A function that takes a float x between a and b, and returns a float representing f(x).
    :param a: The lower limit of integration.
    :param b: The upper limit of integration.
    :param n: A positive integer representing the precision of the result, where higher is more precise and vice versa.

    :return: An approximation of the integral of f between a and b.
    """

    result = (f(a) + f(b)) / 2
    for m in range(1, n):
        result += f(a + (m / n) * (b - a))
    result *= (b - a) / n
    return result


def inverse(f, y, x_lwr=None, x_upr=None, steps=100):
    """
    :param f: An invertible function that takes a float x between a and b, and returns a float representing f(x).
    :param y: A floating point number representing y = f(x), where x is what will be returned.
    :param x_lwr: The lowest x for which f(x) is defined, this parameter is if there is no bound.
    :param x_upr: The highest x for which f(x) is defined, this parameter is None if there is no bound.
    :param steps: How many steps to repeat the bisect method for. Increase for greater answer precision.

    :return: An estimate for the x value that corresponds to f(x) = y, obtained with the bisect method.
    """

    # If no bounds are provided, add them.
    if x_lwr is None or x_upr is None:
        if x_lwr is None and x_upr is None:
            x_lwr, x_upr = -2, 2
        elif x_upr is not None:
            x_lwr = min(-2, x_upr - 1)
        elif x_lwr is not None:
            x_upr = max(2, x_lwr + 1)

        if f(x_lwr) > f(x_upr):
            x_lwr, x_upr = x_upr, x_lwr

        if x_lwr == x_upr:
            exit(f"CustomError: In inverse; f({x_lwr}) = f({x_upr}), inputted function is not invertible.")
        while y < f(x_lwr):
            x_lwr *= 2
        while y > f(x_upr):
            x_upr *= 2

    # Handle errors that arise from inputs.

    if f(x_lwr) > f(x_upr):
        x_lwr, x_upr = x_upr, x_lwr

    if y < f(x_lwr) or y > f(x_upr):
        exit(f"CustomError: In inverse; y={y} out of bounds for [{f(x_lwr)}, {f(x_upr)}] from x_lwr={x_lwr}, x_upr={x_upr}.")
    elif f(x_lwr) == f(x_upr):
        exit(f"CustomError: In inverse; f(x_lwr) = f(x_upr), inputted function is not invertible.")

    # Use bisect method.
    for _ in range(steps):
        x_middle = (x_lwr + x_upr) / 2
        y_estimate = f(x_middle)

        if y_estimate == y:
            return x_middle
        elif y_estimate < y:
            x_lwr = x_middle
        elif y_estimate > y:
            x_upr = x_middle

    return (x_lwr + x_upr) / 2



