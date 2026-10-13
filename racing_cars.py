
import math
import pickle
import random
import pygame
import neat

from neat.nn import FeedForwardNetwork
from car import (
    CENTER, TRACK_RX, TRACK_RY, TRACK_WIDTH, Car
)
from game import Game


CONFIG_FILE = "config-neat.txt"
CHAMPION_FILE = "best_genome.pkl"

FPS = 60
TOTAL_LAPS = 200
START_THETA = math.radians(155)

WHITE = (240, 243, 248)
GREY = (175, 185, 200)
GREEN = (100, 240, 150)
RED = (245, 75, 85)
BLUE = (75, 150, 255)
YELLOW = (255, 210, 70)

RACERS = [
    {
        "name": "RED",
        "color": (245, 65, 75),
        "steering_bias": -0.012,
        "throttle_scale": 1.000,
    },
    {
        "name": "GREEN",
        "color": (55, 220, 125),
        "steering_bias": 0.0,
        "throttle_scale": 0.995,
    },
    {
        "name": "BLUE",
        "color": (65, 145, 255),
        "steering_bias": 0.012,
        "throttle_scale": 1.000,
    },
]


def format_time(seconds):
    if seconds is None:
        return "--:--.--"

    minutes = int(seconds // 60)
    remaining = seconds % 60
    return f"{minutes:02}:{remaining:05.2f}"


def wrap_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def track_theta(car):
    """Ellipse-aware angular position around the track."""
    rx = TRACK_RX - TRACK_WIDTH / 2
    ry = TRACK_RY - TRACK_WIDTH / 2

    dx = (car.x - CENTER[0]) / rx
    dy = (car.y - CENTER[1]) / ry

    return math.atan2(dy, dx)


class VariedBrain:
    """A champion network with a small, controlled driving variation."""

    def __init__(self, network, steering_bias, throttle_scale):
        self.network = network
        self.steering_bias = steering_bias
        self.throttle_scale = throttle_scale

    def activate(self, inputs):
        outputs = list(self.network.activate(inputs))

        outputs[0] += self.steering_bias
        outputs[1] *= self.throttle_scale

        return outputs


def make_racers(champion, config):
    cars = []

    # Three closely spaced grid slots around the shared start line.
    # Small radial offsets create two starting lanes.
    grid = [
        (0.000, -12),
        (-0.055, 12),
        (-0.110, -12),
    ]

    for index, racer in enumerate(RACERS):
        theta_offset, lane_offset = grid[index]
        theta = START_THETA + theta_offset

        rx = TRACK_RX - TRACK_WIDTH / 2 + lane_offset
        ry = TRACK_RY - TRACK_WIDTH / 2 + lane_offset

        x = CENTER[0] + rx * math.cos(theta)
        y = CENTER[1] + ry * math.sin(theta)

        tangent_x = -rx * math.sin(theta)
        tangent_y = ry * math.cos(theta)
        angle = math.atan2(tangent_y, tangent_x)

        # Each car gets its own network instance.
        network = FeedForwardNetwork.create(champion, config)

        brain = VariedBrain(
            network,
            racer["steering_bias"],
            racer["throttle_scale"],
        )

        car = Car(
            x,
            y,
            angle,
            color=racer["color"],
            brain=brain,
        )

        cars.append(car)

    return cars


def new_race_state(cars, now):
    states = []

    for car in cars:
        theta = track_theta(car)

        states.append({
            "name": next(
                racer["name"]
                for racer in RACERS
                if racer["color"] == car.color
            ),
            "color": car.color,
            "laps": 0,
            "best_lap": None,
            "last_laps": [],
            "lap_start": now,
            "previous_theta": theta,
            "left_start": False,
            "progress": 0.0,
            "finished": False,
            "finish_time": None,
        })

    return states


def update_race(cars, states, now):
    """Update lap counts and lap timing independently for each car."""
    rx = TRACK_RX - TRACK_WIDTH / 2
    ry = TRACK_RY - TRACK_WIDTH / 2

    for car, state in zip(cars, states):
        if state["finished"] or not car.alive:
            continue

        theta = track_theta(car)
        previous = state["previous_theta"]

        # Accumulate angular movement for the live leaderboard.
        delta = wrap_angle(theta - previous)
        state["progress"] += delta / (2 * math.pi)

        previous_relative = wrap_angle(
            previous - START_THETA
        )
        current_relative = wrap_angle(
            theta - START_THETA
        )

        # Don't count the starting position as an immediate lap.
        if abs(current_relative) > math.radians(12):
            state["left_start"] = True

        crossed_line = (
                state["left_start"]
                and previous_relative < 0 <= current_relative
        )

        if crossed_line:
            lap_time = (now - state["lap_start"]) / 1000.0

            if lap_time >= 2.0:
                state["laps"] += 1
                state["last_laps"].append(lap_time)

                if (
                        state["best_lap"] is None
                        or lap_time < state["best_lap"]
                ):
                    state["best_lap"] = lap_time

                state["lap_start"] = now
                state["left_start"] = False

                if state["laps"] >= TOTAL_LAPS:
                    state["finished"] = True
                    state["finish_time"] = now / 1000.0

        state["previous_theta"] = theta


def standings(cars, states):
    """Order racers by completed laps, then progress."""
    entries = list(zip(cars, states))

    return sorted(
        entries,
        key=lambda entry: (
            entry[1]["laps"],
            entry[1]["progress"],
        ),
        reverse=True,
    )


def draw_hud(screen, font, small_font, cars, states, race_start, now,
             paused, race_over):
    panel = pygame.Surface((365, 340), pygame.SRCALPHA)
    panel.fill((12, 18, 28, 225))
    screen.blit(panel, (12, 12))

    def write(text, x, y, color=WHITE, use_small=False):
        chosen_font = small_font if use_small else font
        screen.blit(chosen_font.render(text, True, color), (x, y))

    write("RGB GRAND PRIX", 25, 20, GREEN)
    write(f"RACE: {TOTAL_LAPS} LAPS", 25, 55, WHITE, True)
    write(
        f"Elapsed: {format_time((now - race_start) / 1000)}",
        25, 78, GREY, True
    )

    y = 112

    for position, (car, state) in enumerate(
            standings(cars, states), start=1
    ):
        name = state["name"]

        write(
            f"{position}. {name}",
            25, y, state["color"]
        )
        write(
            f"{state['laps']:03}/{TOTAL_LAPS}",
            155, y, WHITE
        )

        current_lap = (now - state["lap_start"]) / 1000.0

        write(
            f"Best {format_time(state['best_lap'])}",
            25, y + 27, WHITE, True
        )
        write(
            f"Now {format_time(current_lap)}",
            190, y + 27, GREY, True
        )

        y += 62

    all_best = [
        state["best_lap"]
        for state in states
        if state["best_lap"] is not None
    ]

    overall_best = min(all_best) if all_best else None

    write(
        f"OVERALL FASTEST: {format_time(overall_best)}",
        25, 300, YELLOW, True
    )

    if paused:
        write("PAUSED - press P to resume", 405, 25, YELLOW)
    elif race_over:
        write("RACE FINISHED - press R to restart", 405, 25, GREEN)


def main():
    with open(CONFIG_FILE, "r", encoding="utf-8") as file:
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            CONFIG_FILE,
        )

    with open(CHAMPION_FILE, "rb") as file:
        champion = pickle.load(file)

    pygame.init()

    game = Game()
    game.show_sensors = False

    screen = pygame.display.get_surface()

    if screen is None:
        game.close()
        pygame.quit()
        raise RuntimeError("Could not access the Pygame display.")

    pygame.display.set_caption("RGB Grand Prix - 200 Lap Race")

    font = pygame.font.SysFont("consolas", 20, bold=True)
    small_font = pygame.font.SysFont("consolas", 15)
    clock = pygame.time.Clock()

    def start_race():
        cars = make_racers(champion, config)
        start_time = pygame.time.get_ticks()
        states = new_race_state(cars, start_time)
        return cars, states, start_time

    cars, states, race_start = start_race()
    paused = False
    race_over = False
    running = True
    frame = 0
    pause_started = None

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

                    elif event.key == pygame.K_s:
                        game.show_sensors = not game.show_sensors

                    elif event.key == pygame.K_p and not race_over:
                        paused = not paused

                        if paused:
                            pause_started = pygame.time.get_ticks()
                        else:
                            # Keep race and lap timers frozen during pause.
                            paused_duration = (
                                    pygame.time.get_ticks() - pause_started
                            )
                            race_start += paused_duration

                            for state in states:
                                state["lap_start"] += paused_duration

                    elif event.key == pygame.K_r:
                        cars, states, race_start = start_race()
                        paused = False
                        race_over = False
                        pause_started = None
                        frame = 0

            if not running:
                break

            now = pygame.time.get_ticks()

            if not paused and not race_over:
                for car, state in zip(cars, states):
                    if not state["finished"]:
                        car.update()

                update_race(cars, states, now)

                # Stop the race once every car has finished or retired.
                if all(
                        state["finished"] or not car.alive
                        for car, state in zip(cars, states)
                ):
                    race_over = True

                frame += 1

            active_cars = sum(
                1 for car, state in zip(cars, states)
                if car.alive and not state["finished"]
            )

            total_fitness = sum(car.fitness for car in cars)

            game.draw(
                cars,
                0,
                active_cars,
                total_fitness,
                frame,
            )

            hud_time = pause_started if paused else now

            draw_hud(
                screen,
                font,
                small_font,
                cars,
                states,
                race_start,
                hud_time,
                paused,
                race_over,
            )

            pygame.display.flip()
            clock.tick(FPS)

    finally:
        game.close()
        pygame.quit()


if __name__ == "__main__":
    main()
