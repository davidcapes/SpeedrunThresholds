import numpy as np

SIMULATORS = [lambda: np.random.exponential(1/2),
              lambda: np.random.exponential(1/3),
              lambda: np.random.exponential(1/0.7)]
W = 1
SIMULATIONS = 100000
Q1 = [0, 0.4867, 0.7337, 10000]
Q1 = [0, 0.3, 0.7, 1]


def tester(simulators, q, w, simulations):
    n = len(simulators)
    total_time = 0

    # Simulate different runs.
    for _ in range(simulations):
        run_time = 0
        task = 1
        while True:

            # Quit or complete.
            if task > n or run_time > q[task - 1]:
                total_time += run_time
                if task > n and run_time <= w:
                    break
                task = 1
                run_time = 0

            # Do current task.
            run_time += simulators[task - 1]()
            task += 1

    return total_time / simulations


print(tester(SIMULATORS, Q1, W, SIMULATIONS))

