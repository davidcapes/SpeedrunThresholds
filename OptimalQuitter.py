import numpy as np
from numba import njit, prange


@njit
def index_table(index_float, table):

    if not 0 <= index_float <= len(table) - 1:
        index_float = min(index_float, len(table) - 1)
        index_float = max(index_float, 0)
        print("Bounding issue with table indexing.")

    r = index_float - np.floor(index_float)
    return (1 - r) * table[int(np.floor(index_float))] + r * table[int(np.ceil(index_float))]


def create_probability_tables(pdfs, bins, x_min, x_max, extra=100):

    # Initialize.
    tasks = len(pdfs)
    bin_width = (x_max - x_min) / bins
    f_table = np.zeros((tasks, bins + 1))
    F_table = np.zeros((tasks, bins + 1))

    # Fill Tables.
    for k in range(0, tasks):
        f = pdfs[k]

        f_table[k][0] = f(x_min)
        F_table[k][0] = 0
        for i in range(1, bins + 1):
            f_table[k][i] = f(x_min + i * bin_width)

            F_table[k][i] = F_table[k][i - 1]
            for j in range(0, extra + 1):
                x = x_min + bin_width * (i - 1 + j / extra)
                F_table[k][i] = F_table[k][i] + f(x) / (extra + 1)
        F_table[k] *= bin_width

    return f_table, F_table


@njit(parallel=True)
def score_z(f_table, F_table, means, w, z, print_warnings=False):

    # Return values for if inversion fails
    lwr_inv_fail, upr_inv_fail = False, False

    # Initialize table parameters.
    tasks = len(F_table)
    bins = len(F_table[0]) - 1

    # Initialize dynamic tables tables.
    q_table = np.full(tasks + 1, w, dtype=np.float64)
    q_table[0] = 0

    s_table_prev = np.zeros(bins + 1, dtype=np.float64)
    s_table_new = np.zeros(bins + 1, dtype=np.float64)

    # Iterate tasks.
    for k in range(tasks - 1, -1, -1):
        q = q_table[k + 1]
        q_index = q * bins / w

        # Update S-Tables.
        for i in prange(len(s_table_prev)):
            s_table_prev[i] = s_table_new[i]
        s_table_new.fill(0)

        for i in prange(0, int(np.floor(q_index)) + 1):

            # Add win factor.
            for j in range(1, int(np.floor(q_index)) - i + 1):
                f1 = s_table_prev[i + j - 1] * f_table[k][j - 1]
                f2 = s_table_prev[i + j] * f_table[k][j]
                s_table_new[i] += (w / bins) * (f1 + f2) / 2

            width = (q_index - np.floor(q_index)) * (w / bins)
            f1 = s_table_prev[int(np.floor(q_index))]
            f2 = index_table(q_index, s_table_prev)
            f1 *= f_table[k][int(np.floor(q_index)) - i]
            f2 *= index_table(q_index - i, f_table[k])
            s_table_new[i] += width * (f1 + f2) / 2

            # Add loss factor.
            s_table_new[i] += (1 - index_table(q_index - i, F_table[k])) * z

            s_table_new[i] += means[k]

        # Add index to allow interpolation at q.
        if np.ceil(q) > np.floor(q):
            r = q_index - np.floor(q_index)
            s_table_new[int(np.ceil(q_index))] = (z - (1 - r) * s_table_new[int(np.floor(q_index))]) / r

        # Quitting thresholds.
        if k > 0:
            if s_table_new[0] >= z:
                q_table[k] = (s_table_new[bins] < z) * w
                if print_warnings:
                    print("Bounding issue with inversion (Lower).")
                lwr_inv_fail = True
                break
            else:
                found = False
                for i in range(1, int(np.ceil(q_index)) + 1):
                    if s_table_new[i] >= z:
                        r = (s_table_new[i] - z) / (s_table_new[i] - s_table_new[i - 1])
                        q_table[k] = (i - r) * w / bins
                        found = True
                        break
                if not found:
                    if print_warnings:
                        print("Bounding issue with inversion (Upper).")
                    upr_inv_fail = True
                    break

    # Results.
    z_new = s_table_new[0]
    return z_new, q_table, lwr_inv_fail, upr_inv_fail


def get_q(pdfs, w, bins, means, iterations=25, multiplier=15.0, exp_lim=1000):

    # Choose initial location.
    f_table, F_table = create_probability_tables(pdfs, bins, 0, w)
    z_new, q_table, lwr, upr = score_z(f_table, F_table, means, w, multiplier * w)
    if bins > 100:
        q_table, z_new = get_q(pdfs, w, 100, means, iterations)
    z_lwr, z_upr = z_new, z_new

    # Exponential bound search.
    z_new, q_table, lwr, upr = score_z(f_table, F_table, means, w, z_lwr)
    for _ in range(exp_lim):
        if z_new >= z_lwr and not lwr:
            break
        z_upr = z_lwr
        z_lwr /= 2
        z_new, q_table, lwr, upr = score_z(f_table, F_table, means, w, z_lwr)
    for _ in range(exp_lim):
        if z_new <= z_upr and not upr:
            break
        z_lwr = z_upr
        z_upr *= 2
        z_new, q_table, lwr, upr = score_z(f_table, F_table, means, w, z_upr)

    # Bisect method.
    for i in range(iterations):
        mid = (z_lwr + z_upr) / 2
        z_new, q_table, lwr, upr = score_z(f_table, F_table, means, w, mid)

        if z_new == mid:
            break
        elif z_new >= mid:
            z_lwr = mid
        elif z_new <= mid:
            z_upr = mid
    return q_table, z_new


pdfs = [lambda x: 2 * np.exp(-2 * x), lambda x: 3 * np.exp(-3 * x), lambda x: 0.7 * np.exp(-0.7 * x)]
w = 1
bins = 1000
means = [1/2, 1/3, 1/0.7]

q, z = get_q(pdfs, w, bins, means)
print("Restart Thresholds:", list(np.round(q, 3)))
print("Expected Time:", np.round(z, 3))
