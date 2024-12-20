import pygame
import random
import csv
import time

# Initialize Pygame
pygame.init()

# Game setup
GOAL_SCORE = 15
TASK_FUNCTIONS = [
    lambda: random.gauss(5, 2),
    lambda: random.uniform(2, 10),
    lambda: random.triangular(1, 10, 4)
]
MID_TASK_RESTARTING = True
PLAY_SPEED = 1 * MID_TASK_RESTARTING + 5 * (not MID_TASK_RESTARTING)

# Screen setup
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Task Game")
font = pygame.font.Font(None, 36)

# Game state
current_task = 1
current_score = 0.0
total_score = 0.0
completion_amount = 0
score_data = []

# Task state
task_in_progress = False
task_start_time = 0.0
current_task_score = 0.0
task_completion_time = 0.0


def draw_text(text, y):
    surface = font.render(text, True, (255, 255, 255))
    screen.blit(surface, (10, y))


def save_data():
    filename = f"GameData/game_data_{time.time()}.csv"

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Task Number', 'Task Score'])
        writer.writerows(score_data)

    print(f"Data saved to {filename}")


def play_game():
    global current_task, current_score, total_score, completion_amount
    global task_in_progress, task_start_time, current_task_score, task_completion_time

    running = True
    clock = pygame.time.Clock()

    while running:
        current_time = time.time()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                save_data()

            if event.type == pygame.MOUSEBUTTONDOWN:

                if event.button == 3 and (MID_TASK_RESTARTING or not task_in_progress):  # Right click - restart
                    if task_in_progress:
                        total_score += min(PLAY_SPEED * (current_time - task_start_time), current_task_score)
                    current_task = 1
                    current_score = 0.0
                    task_in_progress = False

                elif event.button == 1 and not task_in_progress:  # Left click - attempt task
                    current_task_score = TASK_FUNCTIONS[current_task - 1]()
                    task_in_progress = True
                    score_data.append((current_task, current_task_score))
                    task_start_time = current_time

        # Task completion logic
        if task_in_progress and (PLAY_SPEED * (current_time - task_start_time) >= current_task_score):
            current_score += current_task_score
            total_score += current_task_score
            current_task += 1
            task_in_progress = False

            if current_task > len(TASK_FUNCTIONS):
                if current_score < GOAL_SCORE:
                    completion_amount += 1
                current_task = 1
                current_score = 0

        # Drawing
        screen.fill((0, 0, 0))
        draw_text(f"Goal (G): {GOAL_SCORE}", 10)
        draw_text(f"Task: {current_task}", 50)
        draw_text(f"Current score: {current_score:.2f}", 90)
        draw_text(f"Total score: {total_score:.2f}", 130)
        draw_text(f"Completions: {completion_amount}", 170)

        if task_in_progress:
            draw_text(f"Task in progress: {PLAY_SPEED * (current_time - task_start_time):.2f}", 210)
        else:
            draw_text("Click to attempt task", 210)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    play_game()
