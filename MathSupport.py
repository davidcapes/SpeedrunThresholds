import numpy as np
from numba import njit


@njit
def calculate_from_table(x, y_table, x_lwr, x_upr, rounding=8, warning=True):
    """
    :param x: A floating point number.
    :param y_table: A numpy array of y values calculated for a set of equally spaced x values.
    :param x_lwr: The x value that corresponds to first entry in y_table.
    :param x_upr: The x value that corresponds to the last entry in y_table.
    :param rounding: How many decimal places to round the final result to. Used to eliminate floating point errors.
    :param warning: Whether to print warnings about potential errors in the code.

    :return: An estimate of y at the inputted x using the provided y_table and linear interpolation.
    """

    # Ensure x can be calculated from table.
    if len(y_table) == 0:
        print(f"CustomError: In calculate_from_table; y_table is empty.")
        return
    elif x_lwr == x_upr:
        print(f"CustomError: In calculate_from_table; {x_lwr}=x_table_lwr=x_table_upr={str(x_upr)}")
        return
    elif not (x_lwr <= x <= x_upr):
        if warning:
            print("CustomWarning: In calculate_from_table;", float(x), "not in the bounds [", float(x_lwr), ",", float(x_upr), "]")
        x = max(x, x_lwr)
        x = min(x, x_upr)

    # Calculate indexes and bounds for x.
    index_estimate = (len(y_table) - 1) * (x - x_lwr) / (x_upr - x_lwr)
    index_lwr, index_upr = int(np.floor(index_estimate)), int(np.ceil(index_estimate))
    y_lwr, y_upr = y_table[index_lwr], y_table[index_upr]

    # Return calculation.
    if index_lwr == index_upr:
        return np.round(y_lwr, rounding)
    r = (index_estimate - index_lwr) / (index_upr - index_lwr)
    return np.round((1 - r) * y_lwr + r * y_upr, rounding)


@njit
def integral_product(f1_table, f1_lwr, f1_upr, f2_table, f2_min, f2_max, x_min, x_max, bins, offset, warning=True):
    """
    Calculates the integral of g1(x)g2(x+t) from x_min to x_max-t using trapezoidal rule

    Parameters:
    table1, table2: numpy arrays containing the function values
    x_min1, x_max1: domain bounds for g1
    x_min2, x_max2: domain bounds for g2
    x_min, x_max: integration bounds (before shifting by offset)
    bins: number of trapezoids to use
    offset: shift parameter
    """

    # If the shifted interval is empty or outside bounds, return 0
    if x_max - offset <= x_min:
        if x_max - offset < x_min and warning:
            print(f"CustomWarning: In integral_product, lower integration bound ({x_max - offset}) exceeds upper ({x_min}), return value set to 0.")
        return 0.0
    dx = (x_max - offset - x_min) / bins

    # Add outer bound contributions.
    g1_lwr = calculate_from_table(x_min, f1_table, f1_lwr, f1_upr, warning=warning)
    g2_lwr = calculate_from_table(x_min + offset, f2_table, f2_min, f2_max, warning=warning)
    g1_upr = calculate_from_table(x_max - offset, f1_table, f1_lwr, f1_upr, warning=warning)
    g2_upr = calculate_from_table(x_max, f2_table, f2_min, f2_max, warning=warning)
    height_sum = (g1_lwr * g2_lwr + g1_upr * g2_upr) / 2

    # Add inner trapezoid contributions.
    for i in range(1, bins):
        x = x_min + i * dx
        g1 = calculate_from_table(x, f1_table, f1_lwr, f1_upr, warning=warning)
        g2 = calculate_from_table(x + offset, f2_table, f2_min, f2_max, warning=warning)
        height_sum += g1 * g2

    # Scale and return result.
    return height_sum * dx


@njit
def table_inverse(y, y_table, x_lwr, x_upr, warning=True):

    for i in range(1, len(y_table)):
        y1 = y_table[i - 1]
        y2 = y_table[i]

        dx = (x_upr - x_lwr) / (len(y_table) - 1)

        if y1 == y:
            return x_lwr + (i - 1) * dx
        if y2 == y:
            return x_lwr + i * dx
        if y1 < y < y2 or y2 < y < y1:
            alpha = abs(y - y1) / (abs(y - y1) + abs(y - y2))
            return x_lwr + (1 - alpha) * (i - 1) * dx + alpha * i * dx

    if warning:
        print("CustomWarning: table_inverse did not find an inverse value")
    if (y_table[-1] < y_table[0] < y) or (y < y_table[0] < y_table[-1]):
        return x_lwr - 0.00001
    else:
        return x_upr + 0.00001


def integral(f, a, b, n=1000):
    """
    :param f: A function that takes a float x between a and b, and returns a float representing f(x).
    :param a: The lower limit of integration.
    :param b: The upper limit of integration.
    :param n: A positive integer representing the precision of the result, where higher is more precise and vice versa.

    :return: An approximation of the integral of f between a and b.
    """
    if a == b:
        return 0
    elif a > b:
        return -integral(f, b, a, n)

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


def exponential_search_solve(f, lwr=1, upr=1, iterations=25, exp_lim=1000):
    """ Solves the equation f(x) = x, under the assumption that f(x) is bijective.
    :param f:
    :param lwr:
    :param upr:
    :param iterations:
    :param exp_lim:
    :return:
    """

    # Exponential bound search.
    for _ in range(exp_lim):
        if f(lwr) >= lwr:
            break
        upr = lwr
        lwr /= 2
    for _ in range(exp_lim):
        if f(upr) <= upr:
            break
        lwr = upr
        upr *= 2

    # Bisect method.
    for _ in range(iterations):

        mid = (lwr + upr) / 2
        f_mid = f(mid)

        if f_mid == mid:
            break
        elif f_mid >= mid:
            lwr = mid
        elif f_mid <= mid:
            upr = mid

    return (lwr + upr) / 2
