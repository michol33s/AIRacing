
import math
import os
import pickle
import pygame
import neat
import random

from neat.nn import FeedForwardNetwork
from car import CENTER, TRACK_RX, TRACK_RY, TRACK_WIDTH, Car
from game import Game


# Training settings
POPULATION_SIZE = 40
MAX_FRAMES_PER_GENERATION = 2400
MAX_GENERATIONS = 100
SHOW_SENSORS_BY_DEFAULT = False
BASE_SEED = 7
FPS = 60

# Keep the original champion untouched.
BEST_GENOME_FILE = "best_genome_fast.pkl"
FASTEST_LAP_FILE = "fastest_lap_genome.pkl"
FASTEST_LAP_RECORD_FILE = "fastest_lap.txt"

GENERATION = 0
QUIT_REQUESTED = False

BEST_FITNESS = float("-inf")
FASTEST_LAP_FRAMES = None


def make_car(genome, config, index, total_cars):
    """Create a car at a different position on the track."""
    base_theta = math.radians(155)
    theta = base_theta + (2 * math.pi * index / total_cars)

    rx = TRACK_RX - TRACK_WIDTH / 2
    ry = TRACK_RY - TRACK_WIDTH / 2

    x = CENTER[0] + rx * math.cos(theta)
    y = CENTER[1] + ry * math.sin(theta)

    tangent_x = -rx * math.sin(theta)
    tangent_y = ry * math.cos(theta)
    angle = math.atan2(tangent_y, tangent_x)

    palette = [
        (40, 170, 255),
        (255, 105, 90),
        (255, 210, 70),
        (120, 230, 140),
        (205, 130, 255),
        (255, 145, 205),
    ]

    brain = FeedForwardNetwork.create(genome, config)

    return Car(
        x,
        y,
        angle,
        color=palette[index % len(palette)],
        brain=brain,
    )


def save_genome(genome, filename):
    """Save safely using a temporary file."""
    path = os.path.abspath(filename)
    temp_path = path + ".tmp"

    try:
        with open(temp_path, "wb") as file:
            pickle.dump(genome, file)

        os.replace(temp_path, path)

    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def save_if_new_best(genome_id, genome):
    """Save the highest-fitness genome from this training run."""
    global BEST_FITNESS

    if genome.fitness is None or genome.fitness <= BEST_FITNESS:
        return

    save_genome(genome, BEST_GENOME_FILE)
    BEST_FITNESS = genome.fitness

    print("\n*** NEW TRAINING CHAMPION SAVED ***")
    print(f"Generation: {GENERATION}")
    print(f"Genome ID: {genome_id}")
    print(f"Fitness: {BEST_FITNESS:.2f}")
    print(f"File: {os.path.abspath(BEST_GENOME_FILE)}\n")


def save_if_fastest_lap(genome_id, genome, lap_frames):
    """Save the genome that achieved the fastest completed lap."""
    global FASTEST_LAP_FRAMES

    if lap_frames is None or lap_frames <= 0:
        return

    if (
            FASTEST_LAP_FRAMES is not None
            and lap_frames >= FASTEST_LAP_FRAMES
    ):
        return

    FASTEST_LAP_FRAMES = lap_frames
    lap_seconds = lap_frames / FPS

    save_genome(genome, FASTEST_LAP_FILE)

    with open(FASTEST_LAP_RECORD_FILE, "w", encoding="utf-8") as file:
        file.write(f"{lap_seconds:.4f}\n")
        file.write(f"{lap_frames}\n")
        file.write(f"Generation: {GENERATION}\n")
        file.write(f"Genome ID: {genome_id}\n")

    print("\n*** NEW FASTEST LAP! ***")
    print(f"Generation: {GENERATION}")
    print(f"Genome ID: {genome_id}")
    print(f"Lap time: {lap_seconds:.3f} seconds")
    print(f"File: {os.path.abspath(FASTEST_LAP_FILE)}\n")


def evaluate_genomes(genomes, config):
    """Evaluate every genome simultaneously."""
    global QUIT_REQUESTED

    game = Game()
    game.show_sensors = SHOW_SENSORS_BY_DEFAULT

    cars = []
    genome_for_car = {}

    total_cars = len(genomes)

    for index, (genome_id, genome) in enumerate(genomes):
        genome.fitness = 0.0

        car = make_car(
            genome,
            config,
            index,
            total_cars,
        )

        cars.append(car)
        genome_for_car[id(car)] = (genome_id, genome)

    # Track each car's previous personal best lap.
    previous_best_laps = {
        id(car): None for car in cars
    }

    frame = 0
    running = True

    try:
        while (
                running
                and frame < MAX_FRAMES_PER_GENERATION
                and any(car.alive for car in cars)
        ):
            running = game.handle_events()

            if not running:
                QUIT_REQUESTED = True
                break

            for car in cars:
                if not car.alive:
                    continue

                car.update()

                genome_id, genome = genome_for_car[id(car)]
                genome.fitness = car.fitness

                # Check whether this car just set a personal lap record.
                current_best = car.best_lap_frames
                previous_best = previous_best_laps[id(car)]

                if (
                        current_best is not None
                        and (
                        previous_best is None
                        or current_best < previous_best
                )
                ):
                    previous_best_laps[id(car)] = current_best

                    save_if_fastest_lap(
                        genome_id,
                        genome,
                        current_best,
                    )

            best_generation_fitness = max(
                (car.fitness for car in cars),
                default=0.0,
            )

            alive_count = sum(car.alive for car in cars)

            game.draw(
                cars,
                GENERATION,
                alive_count,
                best_generation_fitness,
                frame,
            )

            game.tick(FPS)
            frame += 1

        # Finalize all genomes, including those whose cars crashed.
        for car in cars:
            genome_id, genome = genome_for_car[id(car)]
            genome.fitness = max(0.01, car.fitness)

            save_if_new_best(genome_id, genome)

        print(
            f"\nGeneration {GENERATION} evaluated | "
            f"Frames: {frame}/{MAX_FRAMES_PER_GENERATION} | "
            f"Cars alive: {sum(car.alive for car in cars)} | "
            f"Best fitness this run: {BEST_FITNESS:.2f}"
        )

        if FASTEST_LAP_FRAMES is not None:
            print(
                "Fastest lap this run: "
                f"{FASTEST_LAP_FRAMES / FPS:.3f} seconds"
            )

    finally:
        game.close()

    if QUIT_REQUESTED:
        raise KeyboardInterrupt


def run():
    global GENERATION

    random.seed(BASE_SEED)

    # Load NEAT configuration.
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        "config-neat.txt",
    )

    config.pop_size = POPULATION_SIZE

    # Start a new population. Do not compare new fitness values with
    # the old champion's score because the reward has changed.
    population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())

    def evaluate_with_generation(genomes, neat_config):
        global GENERATION

        GENERATION += 1

        print("\n" + "=" * 55)
        print(f"STARTING GENERATION {GENERATION}")
        print(f"Cars in this generation: {len(genomes)}")
        print("=" * 55)

        evaluate_genomes(genomes, neat_config)

        print(
            f"Generation {GENERATION} complete. "
            f"Best fitness this run: {BEST_FITNESS:.2f}"
        )

    try:
        winner = population.run(
            evaluate_with_generation,
            MAX_GENERATIONS,
        )

        print("\nTraining complete.")
        print("Best genome returned by NEAT:")
        print(winner)

        save_if_new_best("final winner", winner)

        print(f"\nBest fitness this run: {BEST_FITNESS:.2f}")
        print(
            "Fitness champion: "
            f"{os.path.abspath(BEST_GENOME_FILE)}"
        )

        if FASTEST_LAP_FRAMES is not None:
            print(
                f"Fastest lap: {FASTEST_LAP_FRAMES / FPS:.3f} seconds"
            )
            print(
                "Fastest-lap genome: "
                f"{os.path.abspath(FASTEST_LAP_FILE)}"
            )
        else:
            print(
                "No complete lap was recorded during this training run."
            )

    except KeyboardInterrupt:
        print("\nTraining stopped.")

        if BEST_FITNESS > float("-inf"):
            print(f"Best fitness recorded: {BEST_FITNESS:.2f}")
            print(f"Saved genome: {BEST_GENOME_FILE}")

        if FASTEST_LAP_FRAMES is not None:
            print(
                "Fastest lap recorded: "
                f"{FASTEST_LAP_FRAMES / FPS:.3f} seconds"
            )

    finally:
        pygame.quit()


if __name__ == "__main__":
    run()
