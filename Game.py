import pygame
import csv
import time
import numpy as np

# Initialize Pygame
pygame.init()
np.random.seed(501)

# Game setup
from R_Tester import simulator_functions
goal_score = 75
mid_task_restarting = False
play_speed = 2 * mid_task_restarting + 10 * (not mid_task_restarting)
USERNAME = "test_user"

# Screen setup
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Restarting Game")
font = pygame.font.Font(None, 36)

# Game state
current_task = 1
current_score = 0.0
total_score = 0.0
completion_amount = 0
score_data = []

task_in_progress = False
task_start_time = 0.0
current_task_score = 0.0


def draw_text(text, y):
    surface = font.render(text, True, (255, 255, 255))
    screen.blit(surface, (10, y))


def save_data():
    filename = f"GameData/game_data_{USERNAME}_{time.time()}.csv"

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Task Number', 'Task Score'])
        writer.writerows(score_data)

    print(f"Data saved to {filename}")


def play_game():
    global current_task, current_score, total_score, completion_amount
    global task_in_progress, task_start_time, current_task_score

    running = True
    clock = pygame.time.Clock()

    while running:
        current_time = time.time()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                save_data()

            if event.type == pygame.MOUSEBUTTONDOWN:

                if event.button == 3 and (mid_task_restarting or not task_in_progress):  # Right click - restart
                    if task_in_progress:
                        total_score += min(play_speed * (current_time - task_start_time), current_task_score)
                    current_task = 1
                    current_score = 0.0
                    task_in_progress = False

                elif event.button == 1 and not task_in_progress:  # Left click - attempt task
                    current_task_score = simulator_functions[current_task - 1]()
                    task_in_progress = True
                    score_data.append((current_task, current_task_score))
                    task_start_time = current_time

        # Task completion logic
        if task_in_progress and (play_speed * (current_time - task_start_time) >= current_task_score):
            current_score += current_task_score
            total_score += current_task_score
            current_task += 1
            task_in_progress = False

            if current_task > len(simulator_functions):
                if current_score < goal_score:
                    completion_amount += 1
                current_task = 1
                current_score = 0
        current_task_time = task_in_progress * (play_speed * (current_time - task_start_time))

        # Drawing
        screen.fill((0, 0, 0))
        draw_text(f"Goal score: {goal_score}", 10)
        draw_text(f"Task: {current_task}", 50)
        draw_text(f"Current score: {current_score + current_task_time:.2f}", 90)
        draw_text(f"Total score: {total_score + current_task_time:.2f}", 130)
        draw_text(f"Completions: {completion_amount}", 170)

        if task_in_progress:
            draw_text(f"Task in progress", 210)
        else:
            draw_text("Click to attempt task", 210)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    play_game()
