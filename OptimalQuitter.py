import numpy as np

from MathSupport import *


def quitter(w, Z, cdfs, means, bins=200):  # Table Size and Integration Precision.

    # Initialize and define variables, tables, and functions.
    n = len(cdfs)
    t_min = 0

    q_table = np.zeros(n + 1)
    q_table[n] = w

    s_table = np.zeros(bins + 1)
    i_table = np.zeros(bins + 1)
    cdf_integral_table = np.zeros(bins + 1)  # Conditional Expectation Table
    d_table = np.zeros(bins + 1)  # s function derivative table
    t_min, t_max = 0, q_table[n]

    def s(t): return calculate_from_table(t, s_table, t_min, t_max)
    def I(t): return calculate_from_table(t, i_table, t_min, t_max)
    def ds(t): return calculate_from_table(t, d_table, t_min, t_max)
    def cdf_integral(t): return calculate_from_table(t, cdf_integral_table, t_min, t_max)

    # Loop across each task (backwards).
    for i in range(n, 0, -1):
        F = cdfs[i - 1]

        inc = (t_max - t_min) / (len(s_table) - 1)
        t_max = q_table[i]

        # CDF integral table.
        for j in range(1, len(cdf_integral_table)):
            inc = (t_max - t_min) / (len(i_table) - 1)
            t = t_min + j * inc
            cdf_integral_table[j] = cdf_integral_table[j - 1] + (F(t) + F(t - inc)) / 2
        for j in range(1, len(cdf_integral_table)):
            cdf_integral_table[j] *= (t_max - t_min) / j

        # Fill out derivative table for s.
        for j in range(len(d_table)):
            if j == 0:
                d_table[j] = (s_table[j + 1] - s_table[j]) / inc
            elif j == len(s_table) - 1:
                d_table[j] = (s_table[j] - s_table[j - 1]) / inc
            else:
                d_table[j] = (s_table[j + 1] - s_table[j - 1]) / inc

        # Fill out i table.
        for j in range(1, len(i_table) - 1):
            t = t_min + j * inc

            i_table[j] = F(t_max - t) * s(t_max) - F(t_min) * s(t)
            i_table[j] -= integral(lambda q: F(q) * ds(q + t), t_min, t_max - t, len(i_table) - 1 - j)
            i_table[j] /= (t_max - t) - t_min

        # Fill out s table with new values.
        for j in range(len(s_table)):
            t = t_min + j * inc
            s_table[j] = means[i - 1] + (1 - F(t_max - t)) * Z + I(t)

        # Invert s to find new threshold quitting value.
        if i != 1:
            q_table[i - 1] = inverse(lambda t: s(t), Z, t_min, q_table[i])

        print("> Completed sub-step", n - i + 1, "/", n)

    print(">> Result:", list(q_table), "| Comparison", Z, "vs.", round(s_table[t_min], 3))
    return s_table[t_min], q_table


def refine_it(cdfs, w, means, iterations=15, bins=100):
    z_lwr, z_upr = sum(means), sum(means)

    while z_lwr > quitter(w, z_lwr, cdfs, means, bins)[0]:
        z_lwr /= 2
    while z_upr < quitter(w, z_upr, cdfs, means, bins)[0]:
        z_upr *= 2

    q = np.array([w for _ in range(len(cdfs) + 1)])
    z_new = (z_lwr + z_upr) / 2
    for _ in range(iterations):
        z, q = quitter(w, z_new, cdfs, means, bins)
        if z == z_new:
            break
        elif z < z_new:
            z_upr = z_new
        elif z > z_new:
            z_lwr = z_new

        z_new = (z_lwr + z_upr) / 2

    return q
