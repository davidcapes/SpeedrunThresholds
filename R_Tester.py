import numpy as np
from scipy.stats import expon, gamma, norm, uniform
from MathSupport import inverse
import matplotlib.pyplot as plt


def get_simulator_functions(cdfs):
    return tuple([(lambda cdf: lambda: inverse(cdf, np.random.uniform(0, 1)))(cdf)
                  for cdf in cdfs])


def simulator(simulator_functions, restart_scores, simulation_count, mid_task_restarting=False):
    """
    :param simulator_functions:
    :param restart_scores:
    :param simulation_count:
    :param mid_task_restarting:
    :return:
    """

    n = len(simulator_functions)
    total_score = 0.0
    goal_score = restart_scores[-1]

    # Simulate different runs.
    for _ in range(simulation_count):
        task = 1
        current_score = 0.0

        while True:

            current_score += simulator_functions[task - 1]()
            if task == n or current_score >= restart_scores[task - 1]:
                if mid_task_restarting:
                    current_score = min(current_score, restart_scores[task - 1])
                total_score += current_score

                if task == n and current_score < goal_score:
                    break
                task = 1
                current_score = 0.0

            else:
                task += 1
    return total_score / simulation_count

cdfs = (
        lambda x1: norm.cdf(x1, loc=15, scale=np.sqrt(12)),
        lambda x2: uniform.cdf(x2, loc=13, scale=18),
        lambda x3: expon.cdf(x3, scale=20),
        lambda x4: np.arctan((x4 - 13) ** 3 / 2) / np.pi + 1/2,
        lambda x5: (norm.cdf(x5, loc=13, scale=5) + norm.cdf(x5, loc=14, scale=2) + norm.cdf(x5, loc=17, scale=0.5)) / 3,
        lambda x6: gamma.cdf(x6, a=5, scale=3)
        )
simulator_functions = get_simulator_functions(cdfs)

if __name__ == "__main__":
    np.random.seed(501)
    simulation_count = 2000
    restart_scores = np.array([15.03452149, 34.96471806, 41.60327076, 55.71387395, 66.1647081, 70])
    print(simulator(simulator_functions, restart_scores, simulation_count, False))
    restart_scores = np.array(np.array([14.95666667, 35.37333333, 41.44, 55.97666667, 66.10333333, 70.0]))
    print(simulator(simulator_functions, restart_scores, simulation_count, False))
    restart_scores = np.array([15.1431788, 34.89911942, 41.55044778, 55.66557211, 66.14757978, 70.0])
    print(simulator(simulator_functions, restart_scores, simulation_count, False))
