import numpy as np
from numba import njit


@njit(fastmath=True)
def calculate_from_table(x, y_table, x_lwr, x_upr, rounding=10, warning=True):
    """
    :param x: A floating point number.
    :param y_table: A numpy array of y values calculated for a set of equally spaced x values.
    :param x_lwr: The x value that corresponds to first entry in y_table.
    :param x_upr: The x value that corresponds to the last entry in y_table.
    :param rounding: How many decimal places to round the final result to.
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
            print("CustomWarning: In calculate_from_table;", float(x), "not in the bounds [", float(x_lwr), ",",
                  float(x_upr), "]")
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
def integral_product(f1_table, f1_lwr, f1_upr, f2_table, f2_min, f2_max, x_min, x_max, n_bins, offset, warning=True):
    """
    Parameters:
    table1, table2: numpy arrays containing the function values
    x_min1, x_max1: domain bounds for g1
    x_min2, x_max2: domain bounds for g2
    x_min, x_max: integration bounds (before shifting by offset)
    n_bins: number of trapezoids to use
    offset: shift parameter

    :return: the integral of g1(x)g2(x+t) from x_min to x_max-t using the trapezoidal rule
    """

    # If the shifted interval is empty or outside bounds, return 0
    if x_max - offset <= x_min:
        if x_max - offset < x_min and warning:
            print("CustomWarning: In integral_product, lower integration bound", x_max - offset, "exceeds upper", x_min,
                  "return value set to 0.")
        return 0.0
    dx = (x_max - offset - x_min) / n_bins

    # Add outer bound contributions.
    g1_lwr = calculate_from_table(x_min, f1_table, f1_lwr, f1_upr, warning=warning)
    g2_lwr = calculate_from_table(x_min + offset, f2_table, f2_min, f2_max, warning=warning)
    g1_upr = calculate_from_table(x_max - offset, f1_table, f1_lwr, f1_upr, warning=warning)
    g2_upr = calculate_from_table(x_max, f2_table, f2_min, f2_max, warning=warning)
    height_sum = (g1_lwr * g2_lwr + g1_upr * g2_upr) / 2

    # Add inner trapezoid contributions.
    for i in range(1, n_bins):
        x = x_min + i * dx
        g1 = calculate_from_table(x, f1_table, f1_lwr, f1_upr, warning=warning)
        g2 = calculate_from_table(x + offset, f2_table, f2_min, f2_max, warning=warning)
        height_sum += g1 * g2

    # Scale and return result.
    return height_sum * dx


@njit
def table_inverse(y, y_table, x_lwr, x_upr, warning=True):
    """
    :param y: A floating point number.
    :param y_table: A numpy array of y values calculated for a set of equally spaced x values.
    :param x_lwr: The x value that corresponds to first entry in y_table.
    :param x_upr: The x value that corresponds to the last entry in y_table.
    :param warning: Whether to print warnings about potential errors in the code.

    :return: the x value corresponding to y in the y_table. the inverse.
    """

    bin_width = (x_upr - x_lwr) / (len(y_table) - 1)
    for i in range(1, len(y_table)):
        y1 = y_table[i - 1]
        y2 = y_table[i]

        if y1 == y:
            return x_lwr + (i - 1) * bin_width
        if y2 == y:
            return x_lwr + i * bin_width
        if y1 < y < y2 or y2 < y < y1:
            alpha = abs(y - y1) / (abs(y - y1) + abs(y - y2))
            return x_lwr + ((1 - alpha) * (i - 1) + alpha * i) * bin_width

    if warning:
        print("CustomWarning: table_inverse did not find an inverse value")
    if (y_table[-1] < y_table[0] < y) or (y < y_table[0] < y_table[-1]):
        return x_lwr - 0.00000001
    else:
        return x_upr + 0.00000001


def integral(f, a, b, bins):
    """
    :param f: A function that takes in and returns a floating point value.
    :param a: Lower bound of integration.
    :param b: Upper bound of integration.
    :param bins: Number of bins to use for the integration.

    :return: The integral of f(x) between a and b using the trapezoidal rule.
    """

    if a == b:
        return 0
    elif b < a:
        return -integral(f, b, a, bins)

    dx = (b - a) / bins
    height_sum = (f(a) + f(b)) / 2
    for i in range(1, bins):
        x = a + i * dx
        height_sum += f(x)

    return height_sum * dx


def inverse(f, y, iterations=25, max_precision=1e-10):

    # Exponential bound search.
    upr = abs(y)
    lwr = -abs(y)

    for _ in range(iterations):
        if f(lwr) <= y:
            break
        upr = lwr
        lwr *= 2

    for _ in range(iterations):
        if f(upr) >= y:
            break
        lwr = upr
        upr *= 2

    # Bisect method.
    for _ in range(iterations):

        mid = (lwr + upr) / 2
        f_mid = f(mid)

        if abs(f_mid - y) < max_precision:
            break
        elif f_mid < y:
            lwr = mid
        elif f_mid > y:
            upr = mid

    return (lwr + upr) / 2


def fx_equals_x(f, lwr=1, upr=1, iterations=80, precision=1e-8):
    """
    :param f: A non-zero bijective function that takes a float as an argument and returns a float.
    :param lwr: Starting lower bound for the solution.
    :param upr: Starting upper bound for the solution.
    :param iterations: Number of iterations to run using the bisect method.
    :param precision: Precision used for the bisect method.

    :return: A solution to f(x) = x.
    """

    # Exponential bound search.
    for _ in range(iterations):
        if f(lwr) >= lwr:
            break
        upr = lwr
        lwr /= 2
    for _ in range(iterations):
        if f(upr) <= upr:
            break
        lwr = upr
        upr *= 2

    # Bisect method.
    for _ in range(iterations):

        mid = (lwr + upr) / 2
        f_mid = f(mid)

        if abs(f_mid - mid) < precision:
            break
        elif f_mid >= mid:
            lwr = mid
        elif f_mid <= mid:
            upr = mid

    return (lwr + upr) / 2
