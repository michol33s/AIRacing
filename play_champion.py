
import math
import pickle
import pygame
import neat
import random

from collections import deque
from neat.nn import FeedForwardNetwork
from car import CENTER, TRACK_RX, TRACK_RY, TRACK_WIDTH, Car
from game import Game


CONFIG_FILE = "config-neat.txt"
CHAMPION_FILE = "best_genome.pkl"

FPS = 60
HUD_WIDTH = 290


def format_time(seconds):
    if seconds is None:
        return "--:--.--"

    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60

    return f"{minutes:02}:{remaining_seconds:05.2f}"


def angle_difference(angle, reference):
    """Return the signed angular difference in the range [-pi, pi)."""
    return (angle - reference + math.pi) % (2 * math.pi) - math.pi


def draw_hud(
        screen,
        font,
        small_font,
        lap_count,
        current_lap_time,
        best_lap_time,
        last_laps,
):
    panel_height = 230

    panel = pygame.Surface(
        (HUD_WIDTH, panel_height),
        pygame.SRCALPHA,
    )
    panel.fill((15, 20, 30, 225))
    screen.blit(panel, (15, 15))

    white = (240, 243, 248)
    green = (100, 240, 150)
    grey = (170, 180, 195)

    x = 30
    y = 25

    title = font.render("CHAMPION RACE", True, green)
    screen.blit(title, (x, y))
    y += 42

    screen.blit(
        small_font.render("LAP", True, grey),
        (x, y),
    )
    screen.blit(
        font.render(str(lap_count + 1), True, white),
        (175, y - 4),
    )
    y += 34

    screen.blit(
        small_font.render("CURRENT LAP", True, grey),
        (x, y),
    )
    screen.blit(
        font.render(format_time(current_lap_time), True, white),
        (145, y - 4),
    )
    y += 34

    screen.blit(
        small_font.render("BEST LAP", True, grey),
        (x, y),
    )
    screen.blit(
        font.render(format_time(best_lap_time), True, green),
        (145, y - 4),
    )
    y += 38

    screen.blit(
        small_font.render("LAST 4 LAPS", True, grey),
        (x, y),
    )
    y += 24

    if not last_laps:
        screen.blit(
            small_font.render("No completed laps yet", True, white),
            (x, y),
        )
    else:
        # Display the most recent lap first.
        for lap_number, lap_time in reversed(last_laps):
            lap_text = f"Lap {lap_number}:  {format_time(lap_time)}"
            screen.blit(
                small_font.render(lap_text, True, white),
                (x, y),
            )
            y += 23


def main():
    # Load NEAT configuration.
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_FILE,
    )

    # Load the saved champion genome.
    with open(CHAMPION_FILE, "rb") as file:
        champion = pickle.load(file)

    brain = FeedForwardNetwork.create(champion, config)

    pygame.init()

    # Create the champion at random starting point
    theta = random.uniform(0, 2 * math.pi)

    rx = TRACK_RX - TRACK_WIDTH / 2
    ry = TRACK_RY - TRACK_WIDTH / 2

    x = CENTER[0] + rx * math.cos(theta)
    y = CENTER[1] + ry * math.sin(theta)

    tangent_x = -rx * math.sin(theta)
    tangent_y = ry * math.cos(theta)
    angle = math.atan2(tangent_y, tangent_x)

    champion_car = Car(
        x,
        y,
        angle,
        color=(255, 210, 40),
        brain=brain,
    )

    game = Game()
    game.show_sensors = False

    # Use the game's display surface.
    screen = pygame.display.get_surface()

    if screen is None:
        game.close()
        pygame.quit()
        raise RuntimeError("Could not access the Pygame display surface.")

    pygame.display.set_caption("NEAT Champion - Lap Timing")

    font = pygame.font.SysFont("consolas", 21, bold=True)
    small_font = pygame.font.SysFont("consolas", 17)

    clock = pygame.time.Clock()

    # Lap timing state.
    lap_count = 0
    best_lap_time = None
    last_laps = deque(maxlen=4)

    race_start_time = pygame.time.get_ticks()
    lap_start_time = race_start_time

    # The start/finish angle matches the starting angle.
    start_finish_angle = theta

    previous_angle = math.atan2(
        champion_car.y - CENTER[1],
        champion_car.x - CENTER[0],
        )

    # Ignore the initial position; arm timing after the car leaves
    # the start/finish area, preventing an immediate false lap.
    left_start_line = False

    running = True
    frame = 0

    try:
        while running and champion_car.alive:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

                    elif event.key == pygame.K_s:
                        game.show_sensors = not game.show_sensors

            if not running:
                break

            champion_car.update()
            frame += 1

            current_angle = math.atan2(
                champion_car.y - CENTER[1],
                champion_car.x - CENTER[0],
                )

            previous_relative = angle_difference(
                previous_angle,
                start_finish_angle,
            )
            current_relative = angle_difference(
                current_angle,
                start_finish_angle,
            )

            # Arm lap detection once the car has moved away from
            # the starting line.
            if abs(current_relative) > math.radians(12):
                left_start_line = True

            # Detect a forward crossing of the start/finish angle.
            crossed_start_line = (
                    left_start_line
                    and previous_relative < 0 <= current_relative
            )

            now = pygame.time.get_ticks()

            if crossed_start_line:
                lap_time = (now - lap_start_time) / 1000.0

                # Avoid counting tiny accidental crossings as laps.
                if lap_time >= 2.0:
                    lap_count += 1
                    last_laps.append((lap_count, lap_time))

                    if (
                            best_lap_time is None
                            or lap_time < best_lap_time
                    ):
                        best_lap_time = lap_time

                    lap_start_time = now

            previous_angle = current_angle

            # Draw the track and champion car.
            game.draw(
                [champion_car],
                0,
                int(champion_car.alive),
                champion_car.fitness,
                frame,
            )

            # Draw the racing HUD on top of the game.
            current_lap_time = (
                                       pygame.time.get_ticks() - lap_start_time
                               ) / 1000.0

            draw_hud(
                screen,
                font,
                small_font,
                lap_count,
                current_lap_time,
                best_lap_time,
                last_laps,
            )

            pygame.display.flip()
            clock.tick(FPS)

    finally:
        game.close()
        pygame.quit()


if __name__ == "__main__":
    main()
