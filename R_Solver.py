import numpy as np
from MathSupport import *
from numba import njit, prange


def create_probability_tables(pdfs, bins, x_min, x_max, extra_bins=1):

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
        F_table[k][0] = 0.0
        cm_table[k][0] = 0.0

        for i in range(1, bins + 1):
            f_table[k][i] = f(x_min + i * bin_width)

            F_table[k][i] = F_table[k][i - 1]
            cm_table[k][i] = cm_table[k][i - 1]
            for j in range(0, extra_bins + 1):
                x = x_min + bin_width * (i - 1 + j / extra_bins)
                f_x = f(x)
                F_table[k][i] += f_x / (extra_bins + 1)
                cm_table[k][i] += x * f_x / (extra_bins + 1)

        F_table[k] *= bin_width
        cm_table[k] *= bin_width

    return f_table, F_table, cm_table


@njit(parallel=True)
def get_J_table(n_bins_max, i, n, r_table, f_tables, W_tables, b_table=np.array([1.0]), warning=True):

    # Error Handling.
    if i < 1:
        print(f"CustomError: In get_J_Table, i = {i} < 1 must be a positive integer.")
    if len(r_table) < max(1, n):
        print(f"CustomError: In get_J_Table, len(r_table) = {len(r_table)} < max(1, {n}).")
    if len(f_tables) < max(1, n):
        print(f"CustomError: In get_J_Table, len(f_tables) = {len(f_tables)} < max(1, {n}).")
    if len(W_tables) < max(1, n):
        print(f"CustomError: In get_J_Table, len(W_tables) = {len(W_tables)} < max(1, {n}).")

    # Initialize.
    r = r_table[min(i - 1, len(r_table) - 1)]
    r_max = np.max(r_table)
    n_bins = min(n_bins_max, int(n_bins_max * (r / r_max)) + 1)
    J_Table = np.zeros(n_bins + 1, dtype=np.float64)

    # Base Case.
    if i > n:
        return b_table

    # Induction Step.
    else:
        J_Table_prev = get_J_table(n_bins_max, i + 1, n, r_table, f_tables, W_tables, b_table, warning)
        r_prev = r_table[min(i, len(r_table) - 1)]
        for j in prange(n_bins + 1):
            t = j * r / n_bins
            n_calc_bins = min(n_bins_max, int(n_bins_max * (r - t) / r_max) + 1)
            integral_term = integral_product(f_tables[i - 1], 0, r_max, J_Table_prev, 0, r_prev, 0, r,
                                             n_calc_bins, t, warning)
            W_term = calculate_from_table(t, W_tables[i - 1], 0, r_max, warning=warning)
            J_Table[j] = integral_term + W_term

    return J_Table


def get_expected_time(pdfs, means, r_table, n_bins, mid_task_restarting=False, warning=False, extra_bins=50):
    f_tables, F_table, cm_table = create_probability_tables(pdfs, n_bins, 0, r_table[-1], extra_bins)
    n = len(pdfs)

    W_table = np.array([np.array([m]) for m in means])
    if mid_task_restarting:
        W_table = np.zeros((n, n_bins+1), dtype=np.float64)
        r_max = np.max(r_table)
        for i in range(n):
            r = r_table[i]
            for j in range(n_bins + 1):
                t = r * j / n_bins
                ltr_comp = calculate_from_table(r - t, cm_table[i], 0, r_max, warning=warning)
                gtr_comp = (r - t) * (1 - calculate_from_table(r - t, F_table[i], 0, r_max, warning=warning))
                W_table[i][j] = ltr_comp + gtr_comp
    b_table = np.array([0.0])
    J_with_means = get_J_table(n_bins, 1, n, r_table, f_tables, W_table, b_table, warning)[0]

    W_table = np.array([np.zeros(1) for _ in means])
    b_table = np.array([1.0])
    J_without_means = get_J_table(n_bins, 1, n, r_table, f_tables, W_table, b_table, warning)[0]

    return J_with_means / J_without_means


def get_r_table_GD(pdfs, means, w, n_bins=300, precision=0.001, precision_divider=10, mid_task_restarting=False):

    # Score function.
    score = lambda values, n_bins: get_expected_time(pdfs, means, values, n_bins, mid_task_restarting)

    # Initialize values
    n = len(pdfs)
    r_table = np.array([w * (i + 1) / n for i in range(n - 1)] + [w])
    curr_precision = w / precision_divider
    best_score = score(r_table, n_bins)

    while curr_precision >= precision:
        changed = False

        for i in range(n - 1):
            for direction in (-1, 1):

                r = r_table[i] + direction * curr_precision
                r = max(precision, r)
                r = min(r, w)

                r_table_new = r_table.copy()
                r_table_new[i] = r
                for j in range(i):
                    r_table_new[j] = min(r, r_table_new[j])
                for j in range(i + 1, n - 1):
                    r_table_new[j] = max(r, r_table_new[j])

                new_score = score(r_table_new, n_bins)
                if new_score < best_score or (new_score == best_score and r < r_table[i]):
                    np.copyto(r_table, r_table_new)
                    best_score = new_score
                    changed = True

        if not changed:
            curr_precision /= precision_divider

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

        r_table[i - 1] = table_inverse(exp_est, s_table_curr, 0, w, warning=warning)
        s_table_prev[:] = s_table_curr

    # Calculate new expected result.
    r = r_table[0]
    calc_bins = min(n_bins, int(n_bins * r / w) + 1)
    exp_est_new = means[0]
    exp_est_new += (1 - calculate_from_table(r, F_tables[0], 0, w, warning=warning)) * exp_est
    exp_est_new += integral_product(f_tables[0], 0.0, w, s_table_prev, 0.0, w, 0.0, r, calc_bins, 0, warning)

    return exp_est_new, r_table


def get_r_table_inv(pdfs, means, w, n_bins_max, bin_multiplier=10, iterations_max=25, precision=10**(-7),
                    extra_bins=50, warning=False):
    f_tables, F_tables, cm_tables = create_probability_tables(pdfs, n_bins_max, 0, w, extra_bins=extra_bins)
    score = lambda exp_est, n_bins: inv_step(exp_est, w, f_tables, F_tables, means, n_bins, warning)[0]

    exp = w
    n_bins = bin_multiplier
    while n_bins < n_bins_max:
        for _ in range(iterations_max):
            exp_new = score(exp, n_bins)
            if abs(exp_new - exp) < precision:
                break
            exp = exp_new
        n_bins = min(n_bins_max, bin_multiplier * n_bins)

    return inv_step(exp, w, f_tables, F_tables, means, n_bins_max, warning)[1]


if __name__ == "__main__":
    pdfs = [lambda x: 2 * np.exp(-2 * x), lambda x: 3 * np.exp(-3 * x), lambda x: 0.7 * np.exp(-0.7 * x)]
    means = np.array([1/2, 1/3, 1/0.7], dtype=np.float64)
    bins = 1000
    r_table = np.array([0.4867, 0.7337, 1.0])
    w = r_table[-1]
