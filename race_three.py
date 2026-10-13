
import math
import os
import pickle

import pygame
import neat

from car import (
    Car,
    WIDTH,
    HEIGHT,
    CENTER,
    TRACK_RX,
    TRACK_RY,
    TRACK_WIDTH,
    INNER_RX,
    INNER_RY,
    BG,
    TRACK_COLOR,
    ROAD_LINE,
    TURQUOISE,
)

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

CONFIG_FILE = "config-neat.txt"
BEST_FILE = "f1_pro_best_genome.pkl"
FASTEST_FILE = "f1_pro_fastest_lap_genome.pkl"

FPS = 60

COLORS = [
    (255, 205, 50),   # Gold
    (60, 190, 255),   # Blue
    (255, 75, 100),   # Red
]

pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("F1 Three Car Championship")
clock = pygame.time.Clock()

font = pygame.font.SysFont("consolas", 20, bold=True)
small_font = pygame.font.SysFont("consolas", 15)

SHOW_SENSORS = False


# --------------------------------------------------
# LOAD SAVED GENOMES
# --------------------------------------------------

def load_genome(filename):
    if not os.path.isfile(filename):
        raise FileNotFoundError(
            f"Missing {filename}. Put it in the same folder as race_three.py."
        )

    with open(filename, "rb") as file:
        return pickle.load(file)


def format_time(seconds):
    if seconds is None:
        return "--:--.---"

    minutes = int(seconds // 60)
    remaining = seconds % 60

    return f"{minutes:02}:{remaining:06.3f}"


# --------------------------------------------------
# TRACK DRAWING
# --------------------------------------------------

def draw_track():
    screen.fill(BG)

    pygame.draw.ellipse(
        screen,
        TRACK_COLOR,
        (
            CENTER[0] - TRACK_RX,
            CENTER[1] - TRACK_RY,
            TRACK_RX * 2,
            TRACK_RY * 2,
        ),
    )

    pygame.draw.ellipse(
        screen,
        BG,
        (
            CENTER[0] - INNER_RX,
            CENTER[1] - INNER_RY,
            INNER_RX * 2,
            INNER_RY * 2,
        ),
    )

    pygame.draw.ellipse(
        screen,
        ROAD_LINE,
        (
            CENTER[0] - TRACK_RX,
            CENTER[1] - TRACK_RY,
            TRACK_RX * 2,
            TRACK_RY * 2,
        ),
        2,
    )

    pygame.draw.ellipse(
        screen,
        ROAD_LINE,
        (
            CENTER[0] - INNER_RX,
            CENTER[1] - INNER_RY,
            INNER_RX * 2,
            INNER_RY * 2,
        ),
        2,
    )

    # Start/finish line at the top of the circuit.
    start_y = CENTER[1] - (TRACK_RY + INNER_RY) / 2

    pygame.draw.line(
        screen,
        (255, 255, 255),
        (CENTER[0] - 12, int(start_y)),
        (CENTER[0] + 12, int(start_y)),
        4,
    )


# --------------------------------------------------
# STARTING GRID
# --------------------------------------------------

def create_racers(config):
    genomes = [
        load_genome(FASTEST_FILE),
        load_genome(BEST_FILE),
        load_genome(FASTEST_FILE),
    ]

    racers = []

    # Staggered grid on the upper-left portion of the oval.
    # The coordinates are derived from the ellipse so the cars
    # begin inside the road instead of outside its boundaries.
    grid_angles = [
        math.radians(225),
        math.radians(225),
        math.radians(225),
    ]

    # Cars start at the same circuit section but in separate
    # lanes, with longitudinal spacing between rows.
    road_rx = (TRACK_RX + INNER_RX) / 2
    road_ry = (TRACK_RY + INNER_RY) / 2

    # Approximate grid locations on the upper-left curve.
    grid_positions = [
        (CENTER[0] - 210, CENTER[1] - 145),
        (CENTER[0] - 250, CENTER[1] - 120),
        (CENTER[0] - 290, CENTER[1] - 95),
    ]

    for i, genome in enumerate(genomes):
        network = neat.nn.FeedForwardNetwork.create(genome, config)

        theta = grid_angles[i]
        car = Car(theta=theta)

        car.x, car.y = grid_positions[i]

        # Tangent of the ellipse at the car's location.
        ellipse_theta = math.atan2(
            (car.y - CENTER[1]) / TRACK_RY,
            (car.x - CENTER[0]) / TRACK_RX,
            )

        tangent_x = -TRACK_RX * math.sin(ellipse_theta)
        tangent_y = TRACK_RY * math.cos(ellipse_theta)

        car.angle = math.atan2(tangent_y, tangent_x)

        car.speed = 0.0
        car.alive = True
        car.fitness = 0.0
        car.completed_laps = 0
        car.lap_frames = 0
        car.best_lap_frames = None

        car.previous_x = car.x
        car.previous_y = car.y
        car.previous_progress = car.track_progress()
        car.progress_since_lap = 0.0
        car.stuck_frames = 0
        car.color = COLORS[i]

        racers.append({
            "name": f"CAR {i + 1}",
            "car": car,
            "network": network,
            "best_lap": None,
            "current_lap_start": pygame.time.get_ticks(),
            "lap_times": [],
            "previous_laps": 0,
            "finished": False,
        })

    return racers


# --------------------------------------------------
# UPDATE RACER
# --------------------------------------------------

def update_racer(racer):
    car = racer["car"]

    if not car.alive:
        return

    inputs = car.sensors()
    inputs += [
        car.speed / car.max_speed,
        math.cos(car.angle - car.track_tangent()),
        ]

    outputs = racer["network"].activate(inputs)

    previous_laps = car.completed_laps
    car.update(outputs)

    if car.completed_laps > previous_laps:
        now = pygame.time.get_ticks()

        lap_seconds = (
                              now - racer["current_lap_start"]
                      ) / 1000.0

        if lap_seconds > 0:
            racer["lap_times"].append(lap_seconds)

            if (
                    racer["best_lap"] is None
                    or lap_seconds < racer["best_lap"]
            ):
                racer["best_lap"] = lap_seconds

        racer["current_lap_start"] = now


# --------------------------------------------------
# LIVE RACE ORDER
# --------------------------------------------------

def race_progress(racer):
    car = racer["car"]

    return (
            car.completed_laps
            + car.progress_since_lap / (2 * math.pi)
    )



def draw_leaderboard(racers, race_seconds):
    ordered = sorted(
        racers,
        key=lambda racer: (
            race_progress(racer),
            racer["car"].fitness,
        ),
        reverse=True,
    )

    panel_bg = (12, 17, 27, 220)
    white = (240, 243, 248)
    grey = (175, 185, 200)
    green = (100, 240, 150)

    margin = 12
    panel_width = 270
    panel_height = 150

    def draw_panel(x, y, width, height, title):
        panel = pygame.Surface((width, height), pygame.SRCALPHA)
        panel.fill(panel_bg)
        screen.blit(panel, (x, y))

        screen.blit(
            font.render(title, True, TURQUOISE),
            (x + 12, y + 10),
        )

    # --------------------------------------------------
    # TOP LEFT: LIVE POSITIONS AND LAPS
    # --------------------------------------------------
    x = margin
    y = margin
    draw_panel(x, y, panel_width, panel_height, "LIVE POSITIONS")

    row_y = y + 43

    for position, racer in enumerate(ordered):
        car = racer["car"]
        index = int(racer["name"][-1]) - 1

        status = "RUN" if car.alive else "OUT"

        line = (
            f"P{position + 1}  {racer['name']}  "
            f"L{car.completed_laps}  {status}"
        )

        screen.blit(
            small_font.render(line, True, COLORS[index]),
            (x + 12, row_y),
        )

        row_y += 29

    # --------------------------------------------------
    # TOP RIGHT: CURRENT LAP TIMES
    # --------------------------------------------------
    x = WIDTH - panel_width - margin
    y = margin
    draw_panel(x, y, panel_width, panel_height, "CURRENT LAP")

    row_y = y + 43

    now = pygame.time.get_ticks()

    for racer in racers:
        index = int(racer["name"][-1]) - 1

        if racer["car"].completed_laps == 0:
            elapsed = (now - race_start_time) / 1000.0
        else:
            elapsed = (now - racer["current_lap_start"]) / 1000.0

        line = f"{racer['name']}: {format_time(elapsed)}"

        screen.blit(
            small_font.render(line, True, COLORS[index]),
            (x + 12, row_y),
        )

        row_y += 29

    # --------------------------------------------------
    # BOTTOM LEFT: PERSONAL BEST LAPS
    # --------------------------------------------------
    x = margin
    y = HEIGHT - panel_height - margin
    draw_panel(x, y, panel_width, panel_height, "PERSONAL BEST")

    row_y = y + 43

    for racer in racers:
        index = int(racer["name"][-1]) - 1
        best = format_time(racer["best_lap"])

        line = f"{racer['name']}: {best}"

        screen.blit(
            small_font.render(line, True, COLORS[index]),
            (x + 12, row_y),
        )

        row_y += 29

    # --------------------------------------------------
    # BOTTOM RIGHT: RACE STATUS AND CONTROLS
    # --------------------------------------------------
    x = WIDTH - panel_width - margin
    y = HEIGHT - panel_height - margin
    draw_panel(x, y, panel_width, panel_height, "RACE STATUS")

    screen.blit(
        small_font.render(
            f"Race time: {format_time(race_seconds)}",
            True,
            white,
        ),
        (x + 12, y + 43),
    )

    active = sum(1 for racer in racers if racer["car"].alive)

    screen.blit(
        small_font.render(
            f"Cars running: {active}/3",
            True,
            green if active else grey,
        ),
        (x + 12, y + 68),
    )

    screen.blit(
        small_font.render("SPACE: Pause  R: Restart", True, white),
        (x + 12, y + 96),
    )

    screen.blit(
        small_font.render("S: Sensors  ESC: Quit", True, white),
        (x + 12, y + 119),
    )



# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------

def main():
    global SHOW_SENSORS

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_FILE,
    )

    racers = create_racers(config)

    running = True
    paused = False
    race_start = pygame.time.get_ticks()
    global race_start_time
    race_start_time = race_start


    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

                    elif event.key == pygame.K_s:
                        SHOW_SENSORS = not SHOW_SENSORS

                    elif event.key == pygame.K_SPACE:
                        paused = not paused

                    elif event.key == pygame.K_r:
                        racers = create_racers(config)
                        race_start = pygame.time.get_ticks()
                        paused = False

            if not paused:
                for racer in racers:
                    update_racer(racer)

            race_seconds = (
                                   pygame.time.get_ticks() - race_start
                           ) / 1000.0

            draw_track()

            for racer in racers:
                racer["car"].draw(screen, SHOW_SENSORS)

            draw_leaderboard(racers, race_seconds)

            screen.blit(
                small_font.render(
                    "SPACE: Pause | R: Restart | S: Sensors | ESC: Quit",
                    True,
                    (230, 230, 240),
                ),
                (15, HEIGHT - 28),
            )

            if paused:
                label = font.render(
                    "PAUSED",
                    True,
                    (255, 220, 80),
                )
                screen.blit(
                    label,
                    label.get_rect(
                        center=(WIDTH // 2, 35)
                    ),
                )

            pygame.display.flip()
            clock.tick(FPS)

    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
