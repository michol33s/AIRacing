import math
import pygame
from car import (
    WIDTH, HEIGHT, CENTER, TRACK_RX, TRACK_RY, TRACK_WIDTH,
    INNER_RX, INNER_RY, point_on_track,
)

GRASS = (42, 112, 62)
ROAD = (76, 80, 86)
WHITE = (235, 235, 235)
DARK = (35, 35, 35)


def draw_track(screen):
    screen.fill(GRASS)
    pygame.draw.ellipse(
        screen, ROAD,
        (CENTER[0] - TRACK_RX, CENTER[1] - TRACK_RY, TRACK_RX * 2, TRACK_RY * 2),
    )
    pygame.draw.ellipse(
        screen, GRASS,
        (CENTER[0] - INNER_RX, CENTER[1] - INNER_RY, INNER_RX * 2, INNER_RY * 2),
    )
    pygame.draw.ellipse(
        screen, WHITE,
        (CENTER[0] - TRACK_RX, CENTER[1] - TRACK_RY, TRACK_RX * 2, TRACK_RY * 2), 4,
    )
    pygame.draw.ellipse(
        screen, WHITE,
        (CENTER[0] - INNER_RX, CENTER[1] - INNER_RY, INNER_RX * 2, INNER_RY * 2), 4,
    )

    # Start/finish marker on the left side of the oval.
    sx, sy = CENTER[0] - TRACK_RX + 8, CENTER[1]
    for i in range(8):
        color = WHITE if i % 2 == 0 else DARK
        pygame.draw.rect(screen, color, (sx, sy - 36 + i * 9, 10, 9))

    # Dashed track centerline.
    for i in range(64):
        t0 = i * 2 * math.pi / 64
        t1 = (i + 0.55) * 2 * math.pi / 64
        pygame.draw.line(screen, (135, 138, 142), point_on_track(t0), point_on_track(t1), 2)


class Game:
    def __init__(self, width=WIDTH, height=HEIGHT):
        pygame.init()
        pygame.display.set_caption("AI Car Racing — 7 Inputs / 2 Outputs")
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 19)
        self.show_sensors = False

    def draw(self, cars, generation, alive_count, best_fitness, frame):
        draw_track(self.screen)
        for car in cars:
            car.draw(self.screen, self.show_sensors)

        lines = [
            f"Generation: {generation}",
            f"Cars alive: {alive_count}/{len(cars)}",
            f"Best fitness this generation: {best_fitness:.1f}",
            f"Frame: {frame}",
            "S: toggle sensors   ESC: quit",
        ]
        for i, line in enumerate(lines):
            image = self.font.render(line, True, WHITE)
            self.screen.blit(image, (18, 15 + i * 26))
        pygame.display.flip()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_s:
                    self.show_sensors = not self.show_sensors
        return True

    def tick(self, fps=60):
        self.clock.tick(fps)

    def close(self):
        pygame.display.quit()
