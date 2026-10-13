
import math
import random
import pickle
import pygame
import neat
import os

from neat.nn import FeedForwardNetwork
from car import CENTER, TRACK_RX, TRACK_RY, TRACK_WIDTH, Car
from game import Game


# Training settings
POPULATION_SIZE = 40
MAX_FRAMES_PER_GENERATION = 2400
MAX_GENERATIONS = 100
SHOW_SENSORS_BY_DEFAULT = False
BASE_SEED = 7

# Global state
GENERATION = 0
QUIT_REQUESTED = False

# Best-genome tracking
BEST_FITNESS = float("-inf")
BEST_GENOME_FILE = "best_genome.pkl"


def make_car(genome, config, index, total_cars):
    """
    Create a car at a different position around the oval.
    All cars face the forward direction of the track.
    """

    # Spread starting positions evenly around the track.
    base_theta = math.radians(155)
    theta = base_theta + (2 * math.pi * index / total_cars)

    # Track centerline.
    rx = TRACK_RX - TRACK_WIDTH / 2
    ry = TRACK_RY - TRACK_WIDTH / 2

    x = CENTER[0] + rx * math.cos(theta)
    y = CENTER[1] + ry * math.sin(theta)

    # Align the car with the track's forward tangent.
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


def load_existing_best():
    """
    Load the saved champion's fitness if available.
    This helps preserve the best score across restarts.
    """
    global BEST_FITNESS

    if not os.path.isfile(BEST_GENOME_FILE):
        return

    try:
        with open(BEST_GENOME_FILE, "rb") as file:
            saved_genome = pickle.load(file)

        saved_fitness = getattr(saved_genome, "fitness", None)

        if saved_fitness is not None:
            BEST_FITNESS = saved_fitness
            print(
                f"Loaded existing champion fitness: "
                f"{BEST_FITNESS:.2f}"
            )

    except Exception as error:
        print(f"Could not load existing champion: {error}")


def save_if_new_best(genome_id, genome):
    """
    Save a genome immediately when it beats the best fitness.
    Use a temporary file to reduce the risk of partial reads.
    """
    global BEST_FITNESS

    fitness = genome.fitness

    if fitness is None or fitness <= BEST_FITNESS:
        return

    save_path = os.path.abspath(BEST_GENOME_FILE)
    temp_path = save_path + ".tmp"

    try:
        with open(temp_path, "wb") as file:
            pickle.dump(genome, file)

        # Replace the champion file only after writing succeeds.
        os.replace(temp_path, save_path)

        BEST_FITNESS = fitness

        print("\n*** NEW BEST GENOME SAVED ***")
        print(f"Generation: {GENERATION}")
        print(f"Genome ID: {genome_id}")
        print(f"Fitness: {fitness:.2f}")
        print(f"File: {save_path}\n")

    except Exception as error:
        print(f"ERROR saving genome: {error}")
        raise

    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def evaluate_genomes(genomes, config):
    """
    Evaluate all genomes simultaneously.
    Each car has its own brain and fitness score.
    Cars do not collide with one another in the current Car class.
    """
    global QUIT_REQUESTED

    game = Game()
    game.show_sensors = SHOW_SENSORS_BY_DEFAULT

    cars = []
    genome_for_car = {}

    total_cars = len(genomes)

    # Create one car for each genome at a different track position.
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

    frame = 0
    running = True

    try:
        # Continue until all cars crash, time expires, or the user quits.
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
                if car.alive:
                    car.update()

                    genome_id, genome = genome_for_car[id(car)]
                    genome.fitness = car.fitness

            # Display the best current fitness among this generation's cars.
            best_fitness = max(
                (car.fitness for car in cars),
                default=0.0,
            )

            alive_count = sum(car.alive for car in cars)

            game.draw(
                cars,
                GENERATION,
                alive_count,
                best_fitness,
                frame,
            )

            game.tick(60)
            frame += 1

        # Finalize scores and save any new records immediately.
        for car in cars:
            genome_id, genome = genome_for_car[id(car)]

            genome.fitness = max(0.01, car.fitness)
            save_if_new_best(genome_id, genome)

        print(
            f"\nGeneration {GENERATION} evaluated | "
            f"Frames: {frame}/{MAX_FRAMES_PER_GENERATION} | "
            f"Cars still alive: {sum(car.alive for car in cars)} | "
            f"Best saved fitness: {BEST_FITNESS:.2f}"
        )

    finally:
        game.close()

    if QUIT_REQUESTED:
        raise KeyboardInterrupt


def run():
    global GENERATION

    random.seed(BASE_SEED)

    # Preserve the saved champion's score when possible.
    load_existing_best()

    # Load NEAT configuration.
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        "config-neat.txt",
    )

    config.pop_size = POPULATION_SIZE

    # Create the evolutionary population.
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
            f"Best saved fitness: {BEST_FITNESS:.2f}"
        )

    try:
        winner = population.run(
            evaluate_with_generation,
            MAX_GENERATIONS,
        )

        print("\nTraining complete.")
        print("Best genome from NEAT:")
        print(winner)

        print(f"\nBest recorded fitness: {BEST_FITNESS:.2f}")
        print(f"Best genome file: {os.path.abspath(BEST_GENOME_FILE)}")

        # Consider the final winner for saving too.
        save_if_new_best("final winner", winner)

    except KeyboardInterrupt:
        print("\nTraining stopped.")

        if BEST_FITNESS > float("-inf"):
            print(f"Best fitness recorded: {BEST_FITNESS:.2f}")
            print(f"Saved genome: {os.path.abspath(BEST_GENOME_FILE)}")

    finally:
        pygame.quit()


if __name__ == "__main__":
    run()
