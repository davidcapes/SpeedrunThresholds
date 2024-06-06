from scipy.stats import norm

from OptimalQuitter import *


# Baseline quitters:
def quitter_nvr(F_array, w):
    return np.array([w for _ in range(len(F_array) + 1)])


def quitter_evs(F_array, w, i_bins=1000, i_min=-100, i_max=500):
    quit_thresholds = [0]
    for F in F_array:
        prev = quit_thresholds[-1]
        mean = integral(lambda x: 1 - F(x), 0, i_max, i_bins) - integral(F, i_min, 0, i_bins)
        quit_thresholds.append(prev + mean)
    return np.array(quit_thresholds)


# Tests
def tester(simulator_array, w, quitter_array, simulations=1000, mid_task_quit=False, seed=None, min_t=-np.inf):
    if seed is not None:
        np.random.seed(seed)

    mean_score = 0
    n = len(simulator_array)

    for _ in range(simulations):
        task = 1
        run_time = 0

        while True:

            # Calculate time to complete current task.
            task_time = max(simulator_array[task - 1](), min_t)
            if mid_task_quit:
                task_time = task_time
            run_time += task_time

            # Run completed in desired time.
            if task == n and run_time <= w:
                mean_score += run_time
                break

            # Restart run at the current task.
            elif task == n or run_time >= quitter_array[task]:
                if mid_task_quit:
                    run_time = quitter_array[task]

                mean_score += run_time
                run_time = 0
                task = 1

            task += 1

    mean_score /= simulations
    return mean_score


# Main
SEED = 87965742
N = 5
CDFS = tuple([lambda x: norm.cdf(x, 40, 5) for _ in range(N)])
W = 170
SIMULATORS = tuple([lambda: norm.rvs(40, 5) for _ in range(N)])

print(tester(SIMULATORS, W, quitter_evs(CDFS, W), seed=SEED, min_t=0, simulations=1000, mid_task_quit=True))
print(tester(SIMULATORS, W, quitter_nvr(CDFS, W), seed=SEED, min_t=0, simulations=1000, mid_task_quit=True))

MEANS = [40 for _ in range(N)]

Q = refine_it(CDFS, W, MEANS, iterations=15, bins=100)
print(tester(SIMULATORS, W, Q, seed=SEED, min_t=0, simulations=1000, mid_task_quit=True))

