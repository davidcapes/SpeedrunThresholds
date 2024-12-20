import numpy as np


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


if __name__ == "__main__":
    simulator_functions = [lambda: np.random.exponential(1/2),
                           lambda: np.random.exponential(1/3),
                           lambda: np.random.exponential(1/0.7)]
    restart_scores = [0.4867, 0.7337, 1]
    simulation_count = 1000000

    print(simulator(simulator_functions, restart_scores, simulation_count, False))
    print(simulator(simulator_functions, restart_scores, simulation_count, True))
