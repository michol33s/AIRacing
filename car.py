import math
import pygame

WIDTH, HEIGHT = 1200, 800
CENTER = (WIDTH // 2, HEIGHT // 2)
TRACK_RX, TRACK_RY = 390, 270
TRACK_WIDTH = 110
INNER_RX, INNER_RY = TRACK_RX - TRACK_WIDTH, TRACK_RY - TRACK_WIDTH

SENSOR_ANGLES = (-70, -35, 0, 35, 70)
SENSOR_RANGE = 180
CAR_LENGTH, CAR_WIDTH = 26, 15


def wrap_angle(angle):
    """Wrap radians to [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def point_on_track(theta):
    """Return a point on the track centerline."""
    rx = TRACK_RX - TRACK_WIDTH / 2
    ry = TRACK_RY - TRACK_WIDTH / 2
    return CENTER[0] + rx * math.cos(theta), CENTER[1] + ry * math.sin(theta)


def is_on_track(x, y, margin=0.0):
    """Check whether a point is between the outer and inner elliptical edges."""
    dx, dy = x - CENTER[0], y - CENTER[1]
    outer_rx, outer_ry = TRACK_RX + margin, TRACK_RY + margin
    inner_rx, inner_ry = INNER_RX - margin, INNER_RY - margin
    if min(outer_rx, outer_ry, inner_rx, inner_ry) <= 0:
        return False
    outer = (dx / outer_rx) ** 2 + (dy / outer_ry) ** 2
    inner = (dx / inner_rx) ** 2 + (dy / inner_ry) ** 2
    return outer <= 1.0 and inner >= 1.0


def ray_distance(x, y, angle, max_distance=SENSOR_RANGE):
    """March a sensor ray forward until it leaves the track."""
    step = 4
    distance = 0
    while distance < max_distance:
        distance += step
        px = x + math.cos(angle) * distance
        py = y + math.sin(angle) * distance
        if not is_on_track(px, py):
            return min(distance, max_distance)
    return max_distance


def track_tangent_angle(x, y):
    """Approximate the forward tangent of the ellipse at the car's location."""
    rx = TRACK_RX - TRACK_WIDTH / 2
    ry = TRACK_RY - TRACK_WIDTH / 2
    dx = (x - CENTER[0]) / rx
    dy = (y - CENTER[1]) / ry
    theta = math.atan2(dy, dx)
    tx = -rx * math.sin(theta)
    ty = ry * math.cos(theta)
    return math.atan2(ty, tx)


class Car:
    def __init__(self, x, y, angle, color=(40, 170, 255), brain=None):
        self.x = float(x)
        self.y = float(y)
        self.angle = float(angle)
        self.color = color
        self.brain = brain
        self.speed = 0.0
        self.max_speed = 6.0
        self.acceleration = 0.16
        self.brake = 0.22
        self.turn_rate = 0.055
        self.alive = True
        self.distance_travelled = 0.0
        self.frames_alive = 0
        self.fitness = 0.0
        self.sensors = [SENSOR_RANGE] * 5

    def get_inputs(self):
        # Inputs 1-5: five forward wall-distance sensors.
        self.sensors = [
            ray_distance(self.x, self.y, self.angle + math.radians(a))
            for a in SENSOR_ANGLES
        ]
        sensor_inputs = [distance / SENSOR_RANGE for distance in self.sensors]

        # Input 6: normalized speed, from 0 to 1.
        speed_input = max(0.0, min(1.0, self.speed / self.max_speed))

        # Input 7: track tangent direction relative to the car, normalized to [-1, 1].
        tangent = track_tangent_angle(self.x, self.y)
        direction_error = wrap_angle(tangent - self.angle)
        direction_input = max(-1.0, min(1.0, direction_error / math.pi))

        return sensor_inputs + [speed_input, direction_input]

    def update(self):
        if not self.alive or self.brain is None:
            return

        outputs = self.brain.activate(self.get_inputs())
        steering = max(-1.0, min(1.0, float(outputs[0])))
        throttle = max(-1.0, min(1.0, float(outputs[1])))

        # Steering authority increases as the car gains speed.
        self.angle += steering * self.turn_rate * (0.45 + self.speed / self.max_speed)

        if throttle >= 0:
            self.speed = min(self.max_speed, self.speed + self.acceleration * throttle)
        else:
            self.speed = max(0.0, self.speed + self.brake * throttle)

        old_x, old_y = self.x, self.y
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed
        self.distance_travelled += math.hypot(self.x - old_x, self.y - old_y)
        self.frames_alive += 1

        if not is_on_track(self.x, self.y, margin=-3):
            self.alive = False

        tangent = track_tangent_angle(self.x, self.y)
        alignment = math.cos(wrap_angle(tangent - self.angle))
        # Reward moving forward in the track's direction and staying alive.
        self.fitness += max(0.0, self.speed * alignment) + 0.03

    def draw(self, surface, show_sensors=False):
        if not self.alive:
            return
        body = pygame.Surface((CAR_LENGTH, CAR_WIDTH), pygame.SRCALPHA)
        pygame.draw.rect(body, self.color, (0, 0, CAR_LENGTH, CAR_WIDTH), border_radius=4)
        pygame.draw.rect(body, (235, 240, 245), (CAR_LENGTH - 8, 3, 5, CAR_WIDTH - 6), border_radius=2)
        rotated = pygame.transform.rotate(body, -math.degrees(self.angle))
        rect = rotated.get_rect(center=(round(self.x), round(self.y)))
        surface.blit(rotated, rect)

        if show_sensors:
            for sensor_angle, distance in zip(SENSOR_ANGLES, self.sensors):
                end = (
                    self.x + math.cos(self.angle + math.radians(sensor_angle)) * distance,
                    self.y + math.sin(self.angle + math.radians(sensor_angle)) * distance,
                )
                pygame.draw.line(surface, (255, 220, 70), (self.x, self.y), end, 1)
