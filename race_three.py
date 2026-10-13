
import math
import os
import pickle
import pygame
import neat

from car import (
    Car, WIDTH, HEIGHT, TRACK_WIDTH, TRACK_CENTERLINE,
    GRASS, TRACK_COLOR, ROAD_LINE, CENTER_LINE
)

FAST_FILE = "f1_pro_fastest_lap_genome.pkl"
BEST_FILE = "f1_pro_best_genome.pkl"
CONFIG_FILE = "config-neat.txt"

FPS = 60
TARGET_LAPS = 200
COLORS = [(0, 255, 210), (255, 90, 90), (255, 210, 60)]
WHITE = (240, 240, 240)
YELLOW = (255, 220, 70)
GREEN = (100, 255, 130)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("F1 NEAT - 200 Lap Race")
clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 17)
small = pygame.font.SysFont("consolas", 14)

N = len(TRACK_CENTERLINE)
SEGMENTS = [
    math.hypot(
        TRACK_CENTERLINE[(i + 1) % N][0] - TRACK_CENTERLINE[i][0],
        TRACK_CENTERLINE[(i + 1) % N][1] - TRACK_CENTERLINE[i][1]
    ) for i in range(N)
]
CUM = [0.0]
for d in SEGMENTS:
    CUM.append(CUM[-1] + d)
TRACK_LENGTH = CUM[-1]


def label(text, x, y, color=WHITE, f=font):
    screen.blit(f.render(str(text), True, color), (x, y))


def lap_time(frames):
    if frames is None:
        return "--:--.---"
    seconds = frames / FPS
    return f"{int(seconds // 60):02d}:{seconds % 60:06.3f}"


def track_index(car):
    return min(range(N), key=lambda i:
    (TRACK_CENTERLINE[i][0] - car.x) ** 2 +
    (TRACK_CENTERLINE[i][1] - car.y) ** 2)


def distance(car):
    return car.completed_laps * TRACK_LENGTH + CUM[track_index(car)]


def draw_track():
    screen.fill(GRASS)
    pts = [(int(x), int(y)) for x, y in TRACK_CENTERLINE]
    pygame.draw.lines(screen, TRACK_COLOR, True, pts, TRACK_WIDTH)
    for p in pts:
        pygame.draw.circle(screen, TRACK_COLOR, p, TRACK_WIDTH // 2)
    pygame.draw.lines(screen, ROAD_LINE, True, pts, 2)

    # Dashed centerline
    for i in range(N):
        x1, y1 = TRACK_CENTERLINE[i]
        x2, y2 = TRACK_CENTERLINE[(i + 1) % N]
        length = math.hypot(x2 - x1, y2 - y1)
        if length == 0:
            continue
        step = 24
        for start in range(0, int(length), step):
            end = min(start + 12, length)
            a, b = start / length, end / length
            pygame.draw.line(screen, CENTER_LINE,
                             (int(x1 + (x2-x1)*a), int(y1 + (y2-y1)*a)),
                             (int(x1 + (x2-x1)*b), int(y1 + (y2-y1)*b)), 2)


def draw_car(car, color, sensors=False):
    x, y, a = car.x, car.y, car.angle
    dark = tuple(max(0, c - 65) for c in color)

    def point(px, py):
        return (int(x + px * math.cos(a) - py * math.sin(a)),
                int(y + px * math.sin(a) + py * math.cos(a)))

    def poly(points, c):
        pygame.draw.polygon(screen, c, [point(px, py) for px, py in points])

    if sensors:
        for offset in (-math.pi/2, -math.pi/4, 0, math.pi/4, math.pi/2):
            end = (int(x + 150 * math.cos(a + offset)),
                   int(y + 150 * math.sin(a + offset)))
            pygame.draw.line(screen, GREEN, (int(x), int(y)), end, 1)

    # Wings and tyres
    poly([(-31,-12),(-25,-15),(-25,15),(-31,12)], dark)
    poly([(25,-15),(32,-13),(32,13),(25,15)], dark)
    for tx, ty in [(-19,-15),(-19,15),(18,-15),(18,15)]:
        poly([(tx-5,ty-3),(tx+5,ty-3),(tx+5,ty+3),(tx-5,ty+3)], (20,20,23))

    # Sidepods and body
    poly([(-13,-12),(12,-12),(19,-7),(15,-4),(-13,-5)], dark)
    poly([(-13,12),(12,12),(19,7),(15,4),(-13,5)], dark)
    poly([(-25,-5),(-15,-8),(10,-7),(26,-3),(30,0),
          (26,3),(10,7),(-15,8),(-25,5)], color)

    # Cockpit, helmet and halo
    poly([(-8,-4),(9,-4),(15,-2),(15,2),(9,4),(-8,4),(-12,0)],
         (28,31,37))
    pygame.draw.circle(screen, WHITE, point(2, 0), 3)
    pygame.draw.lines(screen, (185,185,190), False,
                      [point(-7,-5),point(-2,-2),point(7,-2)], 2)
    pygame.draw.lines(screen, (185,185,190), False,
                      [point(-7,5),point(-2,2),point(7,2)], 2)


def make_car(i):
    start = i * 8
    car = Car(2 * math.pi * start / N)
    car.start_index = start
    car.race_frames = 0
    car.race_color = COLORS[i]
    car.race_finished = False
    return car


def respawn(car, i):
    fresh = Car(2 * math.pi * car.start_index / N)
    fresh.completed_laps = car.completed_laps
    fresh.best_lap_frames = car.best_lap_frames
    fresh.race_frames = car.race_frames
    fresh.start_index = car.start_index
    fresh.race_color = COLORS[i]
    return fresh


def load_brain():
    folder = os.path.dirname(os.path.abspath(__file__))
    path = next((os.path.join(folder, f) for f in
                 (FAST_FILE, BEST_FILE)
                 if os.path.isfile(os.path.join(folder, f))), None)
    if path is None:
        raise FileNotFoundError("Train first: no saved genome found.")

    with open(path, "rb") as f:
        genome = pickle.load(f)

    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        os.path.join(folder, CONFIG_FILE)
    )
    if config.genome_config.num_inputs != 7 or config.genome_config.num_outputs != 2:
        raise ValueError("NEAT config must have 7 inputs and 2 outputs.")

    print("Loaded:", path)
    return neat.nn.FeedForwardNetwork.create(genome, config)


def race():
    net = load_brain()
    cars = [make_car(i) for i in range(3)]
    show_sensors = finished = False
    overall_best = None
    last_gap_update = pygame.time.get_ticks()
    gaps = [0.0] * 3
    running = True

    while running:
        clock.tick(FPS)
        now = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_s:
                    show_sensors = not show_sensors

        if not running:
            break

        if not finished:
            for i, car in enumerate(cars):
                car.race_frames += 1
                if car.alive:
                    inputs = list(car.sensors()) + [
                        car.speed / car.max_speed,
                        math.cos(car.angle - car.track_tangent())
                    ]
                    car.update(net.activate(inputs))

                if not car.alive:
                    cars[i] = respawn(car, i)
                    continue

                if car.best_lap_frames is not None:
                    if overall_best is None or car.best_lap_frames < overall_best:
                        overall_best = car.best_lap_frames

            finished = all(c.completed_laps >= TARGET_LAPS for c in cars)

        # Recalculate displayed gaps every second
        if now - last_gap_update >= 1000:
            leader_dist = max(distance(c) for c in cars)
            gaps = [leader_dist - distance(c) for c in cars]
            last_gap_update = now

        ranked = sorted(range(3), key=lambda i: distance(cars[i]), reverse=True)

        draw_track()
        for i, car in enumerate(cars):
            draw_car(car, COLORS[i], show_sensors)

        # Race leaderboard
        pygame.draw.rect(screen, (15,15,20), (10,10,330,165))
        label("F1 NEAT - 200 LAP RACE", 20, 16, YELLOW)
        for pos, i in enumerate(ranked):
            car = cars[i]
            y = 48 + pos * 39
            label(f"P{pos+1} Car {i+1}", 20, y, COLORS[i])
            label(f"{car.completed_laps}/{TARGET_LAPS} laps", 145, y)
            label(lap_time(car.best_lap_frames), 145, y+17, WHITE, small)

        # Gaps panel, refreshed once per second
        pygame.draw.rect(screen, (15,15,20), (WIDTH-220,10,210,115))
        label("GAPS TO LEADER", WIDTH-210, 17, YELLOW)
        for pos, i in enumerate(ranked):
            gap_text = "LEADER" if pos == 0 else f"+{gaps[i]:.1f} m"
            label(f"Car {i+1}: {gap_text}", WIDTH-210, 45+pos*23, COLORS[i], small)

        label(f"Overall best: {lap_time(overall_best)}", 12, HEIGHT-43, GREEN)
        label("S: Sensors  |  ESC: Exit", 12, HEIGHT-23, WHITE, small)

        if finished:
            label("RACE FINISHED - ESC TO EXIT",
                  WIDTH//2-155, HEIGHT//2, YELLOW)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    race()
