
import argparse
import math
import os
import pickle
import random

import pygame
import neat

from car import (
    Car, WIDTH, HEIGHT, CENTER,
    TRACK_RX, TRACK_RY, INNER_RX, INNER_RY,
    BG, TRACK_COLOR, ROAD_LINE, TURQUOISE,
)

CONFIG_FILE = "config-neat.txt"
BEST_GENOME_FILE = "f1_pro_best_genome.pkl"
FASTEST_LAP_FILE = "f1_pro_fastest_lap_genome.pkl"
FASTEST_RECORD_FILE = "f1_pro_fastest_lap.txt"

FPS = 60
MAX_FRAMES = 2400
MAX_GENERATIONS = 100
POPULATION_SIZE = 40

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("F1 Pro NEAT Racer")
clock = pygame.time.Clock()

font = pygame.font.SysFont("consolas", 20)
small_font = pygame.font.SysFont("consolas", 15)

show_sensors = False
generation = 0
fastest_lap_seconds = float("inf")
best_fitness = float("-inf")


def draw_track():
    screen.fill(BG)

    pygame.draw.ellipse(
        screen, TRACK_COLOR,
        (CENTER[0] - TRACK_RX, CENTER[1] - TRACK_RY,
         TRACK_RX * 2, TRACK_RY * 2)
    )
    pygame.draw.ellipse(
        screen, BG,
        (CENTER[0] - INNER_RX, CENTER[1] - INNER_RY,
         INNER_RX * 2, INNER_RY * 2)
    )
    pygame.draw.ellipse(
        screen, ROAD_LINE,
        (CENTER[0] - TRACK_RX, CENTER[1] - TRACK_RY,
         TRACK_RX * 2, TRACK_RY * 2), 2
    )
    pygame.draw.ellipse(
        screen, ROAD_LINE,
        (CENTER[0] - INNER_RX, CENTER[1] - INNER_RY,
         INNER_RX * 2, INNER_RY * 2), 2
    )

    sx = CENTER[0]
    sy = CENTER[1] - (TRACK_RY + INNER_RY) // 2
    pygame.draw.line(
        screen, (255, 255, 255),
        (sx, sy - 10), (sx, sy + 10), 4
    )


def draw_text(text, x, y, color=(235, 240, 245), font_obj=font):
    screen.blit(font_obj.render(text, True, color), (x, y))


def make_starting_grid(count):
    """Evenly spaced positions with a random offset and shuffled assignment."""
    offset = random.uniform(0, 2 * math.pi)
    angles = [
        (offset + 2 * math.pi * i / count) % (2 * math.pi)
        for i in range(count)
    ]
    random.shuffle(angles)
    return angles


def load_config():
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        CONFIG_FILE
    )

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing {CONFIG_FILE}; place it beside main.py."
        )

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        path,
    )
    config.pop_size = POPULATION_SIZE
    return config


def save_if_fastest_lap(genome, car):
    global fastest_lap_seconds

    if car.best_lap_frames is None:
        return

    lap_seconds = car.best_lap_frames / FPS

    if lap_seconds < fastest_lap_seconds:
        fastest_lap_seconds = lap_seconds

        with open(FASTEST_LAP_FILE, "wb") as file:
            pickle.dump(genome, file)

        with open(FASTEST_RECORD_FILE, "w", encoding="utf-8") as file:
            file.write("F1 Pro fastest lap\n")
            file.write(f"Lap time: {lap_seconds:.4f} seconds\n")
            file.write(f"Frames: {car.best_lap_frames}\n")
            file.write(f"Generation: {generation}\n")

        print(f"New fastest lap: {lap_seconds:.4f}s")


def eval_genomes(genomes, config):
    global generation, best_fitness, show_sensors
    generation += 1

    networks = []
    cars = []
    genome_list = []

    grid = make_starting_grid(len(genomes))

    for index, (genome_id, genome) in enumerate(genomes):
        genome.fitness = 0.0
        networks.append(
            neat.nn.FeedForwardNetwork.create(genome, config)
        )
        cars.append(Car(theta=grid[index]))
        genome_list.append(genome)

    for frame in range(MAX_FRAMES):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    raise SystemExit
                if event.key == pygame.K_s:
                    show_sensors = not show_sensors

        active = 0

        for i, car in enumerate(cars):
            if not car.alive:
                continue

            inputs = car.sensors()
            inputs += [
                car.speed / car.max_speed,
                math.cos(car.angle - car.track_tangent()),
                ]

            outputs = networks[i].activate(inputs)
            car.update(outputs)
            genome_list[i].fitness = car.fitness

            save_if_fastest_lap(genome_list[i], car)

            if car.alive:
                active += 1

        draw_track()

        for car in cars:
            if car.alive:
                car.draw(screen, show_sensors)

        best_this_gen = max(
            (car.fitness for car in cars), default=0.0
        )

        draw_text("F1 PRO | NEAT TRAINING", 20, 15, TURQUOISE)
        draw_text(f"Generation: {generation}", 20, 45)
        draw_text(f"Frame: {frame}/{MAX_FRAMES}", 20, 75)
        draw_text(f"Cars running: {active}/{len(cars)}", 20, 105)
        draw_text(f"Best fitness: {best_this_gen:.1f}", 20, 135)

        record = (
            f"Fastest lap: {fastest_lap_seconds:.2f}s"
            if math.isfinite(fastest_lap_seconds)
            else "Fastest lap: --"
        )
        draw_text(record, 20, 165, TURQUOISE)
        draw_text(
            "S: sensors | ESC: quit",
            20, HEIGHT - 30, (190, 200, 215), small_font
        )

        pygame.display.flip()
        clock.tick(FPS)

        if active == 0:
            break

    for genome in genome_list:
        if genome.fitness is not None and genome.fitness > best_fitness:
            best_fitness = genome.fitness
            with open(BEST_GENOME_FILE, "wb") as file:
                pickle.dump(genome, file)


def play_genome(which):
    global show_sensors

    filename = (
        FASTEST_LAP_FILE if which == "fastest"
        else BEST_GENOME_FILE
    )

    if not os.path.isfile(filename):
        raise FileNotFoundError(
            f"{filename} not found. Train the racer first."
        )

    config = load_config()
    with open(filename, "rb") as file:
        genome = pickle.load(file)

    network = neat.nn.FeedForwardNetwork.create(genome, config)
    car = Car()
    frame = 0
    pygame.display.set_caption(f"F1 Pro Playback: {which}")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_s:
                    show_sensors = not show_sensors
                elif event.key == pygame.K_r:
                    car = Car()
                    frame = 0

        if car.alive:
            inputs = car.sensors()
            inputs += [
                car.speed / car.max_speed,
                math.cos(car.angle - car.track_tangent()),
                ]
            car.update(network.activate(inputs))

        draw_track()
        car.draw(screen, show_sensors)

        draw_text(f"PLAYBACK: {which.upper()}", 20, 15, TURQUOISE)
        draw_text(f"Speed: {car.speed:.2f}", 20, 45)
        draw_text(f"Fitness: {car.fitness:.1f}", 20, 75)
        draw_text(f"Laps: {car.completed_laps}", 20, 105)
        draw_text(f"Frame: {frame}", 20, 135)

        draw_text(
            "S: sensors | R: restart | ESC: quit",
            20, HEIGHT - 30, (190, 200, 215), small_font
        )

        if not car.alive:
            draw_text("OFF TRACK - press R to restart", 20, 175,
                      (255, 100, 100))

        pygame.display.flip()
        clock.tick(FPS)
        frame += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--play", choices=("best", "fastest"),
        help="Watch a saved genome instead of training."
    )
    args = parser.parse_args()

    try:
        if args.play:
            play_genome(args.play)
            return

        config = load_config()
        population = neat.Population(config)
        population.add_reporter(neat.StdOutReporter(True))
        population.add_reporter(neat.StatisticsReporter())

        winner = population.run(eval_genomes, MAX_GENERATIONS)

        with open(BEST_GENOME_FILE, "wb") as file:
            pickle.dump(winner, file)

        print("Training complete.")
        print(f"Best genome saved to {BEST_GENOME_FILE}")

        if os.path.isfile(FASTEST_LAP_FILE):
            print(f"Fastest lap genome saved to {FASTEST_LAP_FILE}")
        else:
            print("No complete lap recorded yet.")

    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
