import numpy as np
from MathSupport import *
from numba import njit, prange


def create_probability_tables(pdfs, bins, x_min, x_max, extra_bins=100):

    # Initialize.
    n = len(pdfs)
    bin_width = (x_max - x_min) / bins
    f_table = np.zeros((n, bins + 1), dtype=np.float64)
    F_table = np.zeros((n, bins + 1), dtype=np.float64)
    cm_table = np.zeros((n, bins + 1), dtype=np.float64)

    # Fill Tables.
    for k in range(0, n):
        f = pdfs[k]

        f_table[k][0] = f(x_min)
        F_table[k][0] = 0
        for i in range(1, bins + 1):
            f_table[k][i] = f(x_min + i * bin_width)

            F_table[k][i] = F_table[k][i - 1]
            cm_table[k][i] = cm_table[k][i - 1]

            for j in range(0, extra_bins + 1):
                x = x_min + bin_width * (i - 1 + j / extra_bins)
                F_table[k][i] = F_table[k][i] + f(x) / (extra_bins + 1)
                cm_table[k][i] = cm_table[k][i] + x * f(x) / (extra_bins + 1)

        F_table[k] *= bin_width
        cm_table[k] *= bin_width

    return f_table, F_table, cm_table


@njit(parallel=True)
def get_J_table(n_bins_max, a, b, r_table, f_tables, b_table=np.array([1.0]), warning=True):
    """
    if a > b, then J(a,b,t) = b(t).
    if a <= b, then J(a,b,t) = integral from 0 to r(a) - t of f_a(t)*J(a + 1,b,t) with respect to t.

    Parameters:
        n_bins_max: Number of bins for the full domain
        a, b: Indices for J_{a,b}
        r_table: Array of R(i) values
        f_tables: List of arrays containing f_i values
        b_table: Base case function B(t) values

    Returns:
        Array of J_{a,b}(t) values
    """

    # Error Handling.
    if a < 1 or b < 1:
        raise ValueError(f"CustomError: In get_J_Table, a = {a} < 1 or b = {b} < 1 must be positive integers.")
    if len(r_table) < max(1, b):
        raise ValueError(f"CustomError: In get_J_Table, len(r_table) = {len(r_table)} < max(1, {b}).")
    if len(f_tables) < max(1, b):
        raise ValueError(f"CustomError: In get_J_Table, len(f_table) = {len(r_table)} < max(1, {b}).")

    # Initialize
    r = r_table[min(a - 1, len(r_table) - 1)]
    r_max = np.max(r_table)
    n_bins = min(n_bins_max, int(n_bins_max * (r / r_max)) + 1)
    J_Table = np.zeros(n_bins + 1, dtype=np.float64)

    # Base Case
    if a > b or a == 0:
        for i in prange(n_bins + 1):
            t = i * r / n_bins
            J_Table[i] = calculate_from_table(t, b_table, 0, r_max, warning=warning)

    # Induction Step
    else:
        J_Table_prev = get_J_table(n_bins_max, a + 1, b, r_table, f_tables, b_table, warning)
        r_prev = r_table[min(a, len(r_table) - 1)]
        for i in prange(n_bins + 1):
            t = i * r / n_bins
            calc_bins = min(n_bins_max, int(n_bins_max * (r - t) / r_max) + 1)
            J_Table[i] = integral_product(f_tables[a - 1], 0.0, r_max, J_Table_prev, 0.0, r_prev, 0.0, r, calc_bins, t, warning)

    return J_Table


def get_expected_time(pdfs, means, r_table, bins):
    f_tables, F_tables, cm_tables = create_probability_tables(pdfs, bins, 0, r_table[-1], extra_bins=100)
    J = lambda a: get_J_table(bins, 1, a, r_table, f_tables, warning=False)[0] if a > 0 else 1
    return (means[0] + sum([means[i]*J(i) for i in range(1, len(pdfs))])) / J(len(pdfs))


def get_r_table_GD(pdfs, means, w, precision=0.0001, bins=2500, bin_multiplier=10, precision_divider=10):

    # Score function.
    score = lambda values, bins: get_expected_time(pdfs, means, values, bins)

    # Initialize values
    n = len(pdfs)
    r_table = np.array([w * (i + 1) / n for i in range(n)])
    r_table[-1] = w

    curr_bins = bin_multiplier
    curr_precision = w / precision_divider

    while curr_bins <= bins:
        print(f"Optimizing for {curr_bins} bins...")

        best_score = score(r_table, bins)
        while curr_precision >= precision:
            changed = False

            for i in range(n - 1):
                for direction in [1, -1]:
                    r = r_table[i] + direction * curr_precision

                    if i < n - 1:
                        r = min(r, r_table[i + 1])
                    if i > 0:
                        r = max(r_table[i - 1], r)
                    r = max(precision, r)
                    r = min(w, r)

                    if r != r_table[i]:
                        r_table_new = r_table.copy()
                        r_table_new[i] = r
                        new_score = score(r_table_new, curr_bins)

                        if new_score < best_score:
                            r_table[i] = r
                            best_score = new_score
                            changed = True

            if not changed:
                curr_precision /= precision_divider
        curr_bins *= bin_multiplier

    return r_table


@njit(parallel=True)
def inv_step(exp_est, w, f_tables, F_tables, means, n_bins, warning=True):

    # Initialize Parameters
    n = len(f_tables)
    r_table = np.full(n, w, dtype=np.float64)

    s_table_prev = np.zeros(n_bins + 1, dtype=np.float64)
    s_table_curr = np.zeros(n_bins + 1, dtype=np.float64)

    if exp_est <= 0:
        return np.inf, r_table

    # Loop across sections.
    for i in range(n - 1, 0, -1):
        r = r_table[i]

        # Update S-Table
        for j in prange(min(n_bins, int(n_bins * r_table[i] / w + 1)) + 1):

            t = j * w / n_bins
            calc_bins = min(n_bins, int(n_bins * (r_table[i] - t) / w) + 1)

            s_table_curr[j] = means[i]
            s_table_curr[j] += (1 - calculate_from_table(r - t, F_tables[i], 0, w, warning=warning)) * exp_est
            s_table_curr[j] += integral_product(f_tables[i], 0.0, w, s_table_prev, 0.0, w, 0.0, r, calc_bins, t, warning)

        r_table[i - 1] = table_inverse(exp_est, s_table_curr, 0, w, warning)
        s_table_prev[:] = s_table_curr

    # Calculate new expected result.
    r = r_table[0]
    calc_bins = min(n_bins, int(n_bins * r / w) + 1)
    exp_est_new = means[0]
    exp_est_new += (1 - calculate_from_table(r, F_tables[0], 0, w, warning=warning)) * exp_est
    exp_est_new += integral_product(f_tables[0], 0.0, w, s_table_prev, 0.0, w, 0.0, r, calc_bins, 0, warning)

    return exp_est_new, r_table


def get_r_table_inv(w, pdfs, means, bins, extra_bins=100, warning=False): # Try using less bins for an imprecise result.
    f_tables, F_tables, cm_tables = create_probability_tables(pdfs, bins, 0, w, extra_bins=extra_bins)
    score = lambda exp_est: inv_step(exp_est, w, f_tables, F_tables, means, bins, warning)[0]

    exp = exponential_search_solve(score, w, 2*w)
    return inv_step(exp, w, f_tables, F_tables, means, bins, warning)[1]


# Main
pdfs = [lambda x: 2 * np.exp(-2 * x), lambda x: 3 * np.exp(-3 * x), lambda x: 0.7 * np.exp(-0.7 * x)]
means = np.array([1/2, 1/3, 1/0.7], dtype=np.float64)
bins = 10000
r_table = np.array([0.4867, 0.7337, 1.0])
w = r_table[-1]

print(get_r_table_inv(w, pdfs, means, bins))
