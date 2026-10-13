
import argparse, math, os, pickle, random
import pygame, neat
from car import (
    Car, WIDTH, HEIGHT, TRACK_WIDTH, TRACK_CENTERLINE,
    GRASS, TRACK_COLOR, ROAD_LINE, CENTER_LINE, TURQUOISE
)

CONFIG_FILE = "config-neat.txt"
BEST_FILE = "f1_pro_best_genome.pkl"
FAST_FILE = "f1_pro_fastest_lap_genome.pkl"
FPS, MAX_FRAMES, MAX_GENERATIONS, POP_SIZE = 60, 900, 100, 40

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("F1 Pro NEAT Racer")
clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 18)
small = pygame.font.SysFont("consolas", 14)
show_sensors, generation = False, 0
best_fitness, fastest_lap = float("-inf"), float("inf")


def draw_track():
    screen.fill(GRASS)
    pts = [(int(x),int(y)) for x,y in TRACK_CENTERLINE]
    pygame.draw.lines(screen, TRACK_COLOR, True, pts, TRACK_WIDTH)
    for p in pts:
        pygame.draw.circle(screen, TRACK_COLOR, p, TRACK_WIDTH//2)
    pygame.draw.lines(screen, ROAD_LINE, True, pts, 2)

    spacing, dash, dist = 22, 11, 0
    for i,(x,y) in enumerate(pts):
        nx,ny = pts[(i+1)%len(pts)]
        length = math.hypot(nx-x,ny-y)
        pos = 0
        while pos < length:
            phase = dist % spacing
            step = min(dash-phase if phase < dash else spacing-phase,
                       length-pos)
            if phase < dash and step > 0:
                a,b = pos/length,(pos+step)/length
                pygame.draw.line(screen,CENTER_LINE,
                                 (int(x+(nx-x)*a),int(y+(ny-y)*a)),
                                 (int(x+(nx-x)*b),int(y+(ny-y)*b)),2)
            pos += step
            dist += step

    x,y = pts[0]
    nx,ny = pts[1]
    length = math.hypot(nx-x,ny-y) or 1
    px,py = -(ny-y)/length,(nx-x)/length
    r = TRACK_WIDTH/2
    pygame.draw.line(screen,(255,255,255),
                     (int(x-px*r),int(y-py*r)),
                     (int(x+px*r),int(y+py*r)),5)


def text(value,x,y,color=(235,240,245),f=None):
    screen.blit((f or font).render(str(value),True,color),(x,y))


def load_config():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),CONFIG_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {CONFIG_FILE}")
    cfg = neat.Config(neat.DefaultGenome,neat.DefaultReproduction,
                      neat.DefaultSpeciesSet,neat.DefaultStagnation,path)
    cfg.pop_size = POP_SIZE
    if cfg.genome_config.num_inputs != 7 or cfg.genome_config.num_outputs != 2:
        raise ValueError("Set num_inputs = 7 and num_outputs = 2 in config-neat.txt")
    return cfg


def inputs_for(car):
    return list(car.sensors()) + [
        car.speed/car.max_speed,
        math.cos(car.angle-car.track_tangent())
    ]


def draw_hud(title,frame,active,total,fitness):
    text(title,20,15,TURQUOISE)
    text(f"Generation: {generation}",20,42)
    text(f"Frame: {frame}/{MAX_FRAMES}",20,68)
    text(f"Cars running: {active}/{total}",20,94)
    text(f"Best fitness: {fitness:.1f}",20,120)
    lap = f"{fastest_lap:.2f}s" if math.isfinite(fastest_lap) else "--"
    text(f"Fastest lap: {lap}",20,146,TURQUOISE)
    text("S: sensors | ESC: quit",20,HEIGHT-25,(190,200,215),small)


def handle_events():
    global show_sensors
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            return False
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_ESCAPE:
                return False
            if e.key == pygame.K_s:
                show_sensors = not show_sensors
    return True


def eval_genomes(genomes,config):
    global generation,best_fitness,fastest_lap
    generation += 1
    n = len(genomes)
    grid = [2*math.pi*i/n for i in range(n)]
    random.shuffle(grid)
    pairs = []
    for (_,genome),theta in zip(genomes,grid):
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome,config)
        pairs.append((genome,net,Car(theta)))

    for frame in range(MAX_FRAMES):
        if not handle_events():
            raise SystemExit

        for genome,net,car in pairs:
            if car.alive:
                car.update(net.activate(inputs_for(car)))
                genome.fitness = car.fitness
                if car.best_lap_frames:
                    lap = car.best_lap_frames/FPS
                    if lap < fastest_lap:
                        fastest_lap = lap
                        folder = os.path.dirname(os.path.abspath(__file__))
                        with open(os.path.join(folder,FAST_FILE),"wb") as f:
                            pickle.dump(genome,f)

        draw_track()
        active = 0
        for _,_,car in pairs:
            if car.alive:
                active += 1
                car.draw(screen,show_sensors)

        top = max((g.fitness or 0 for g,_,_ in pairs),default=0)
        draw_hud("F1 PRO | NEAT TRAINING",frame,active,n,top)
        pygame.display.flip()
        clock.tick(FPS)
        if active == 0:
            break

    folder = os.path.dirname(os.path.abspath(__file__))
    for genome,_,_ in pairs:
        if genome.fitness is not None and genome.fitness > best_fitness:
            best_fitness = genome.fitness
            with open(os.path.join(folder,BEST_FILE),"wb") as f:
                pickle.dump(genome,f)


def play(which,config):
    global show_sensors
    folder = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(folder,FAST_FILE if which=="fastest" else BEST_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{os.path.basename(path)} not found; train first.")
    with open(path,"rb") as f:
        genome = pickle.load(f)
    net = neat.nn.FeedForwardNetwork.create(genome,config)
    car,frame = Car(),0

    while handle_events():
        keys = pygame.key.get_pressed()
        if keys[pygame.K_r]:
            car,frame = Car(),0
        if car.alive:
            car.update(net.activate(inputs_for(car)))

        draw_track()
        car.draw(screen,show_sensors)
        text(f"PLAYBACK: {which.upper()}",20,15,TURQUOISE)
        text(f"Speed: {car.speed:.2f}",20,45)
        text(f"Fitness: {car.fitness:.1f}",20,72)
        text(f"Laps: {car.completed_laps}",20,99)
        text(f"Frame: {frame}",20,126)
        text("S: sensors | R: restart | ESC: quit",20,HEIGHT-25,(190,200,215),small)
        if not car.alive:
            text("OFF TRACK - press R to restart",20,160,(255,100,100))
        pygame.display.flip()
        clock.tick(FPS)
        frame += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--play",choices=("best","fastest"))
    args = parser.parse_args()
    try:
        config = load_config()
        if args.play:
            play(args.play,config)
            return
        pop = neat.Population(config)
        pop.add_reporter(neat.StdOutReporter(True))
        pop.add_reporter(neat.StatisticsReporter())
        winner = pop.run(eval_genomes,MAX_GENERATIONS)
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),BEST_FILE)
        with open(path,"wb") as f:
            pickle.dump(winner,f)
        print(f"Training complete. Best genome saved: {path}")
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
