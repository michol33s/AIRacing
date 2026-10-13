
import math
import pygame

WIDTH, HEIGHT = 1200, 800
CENTER = (WIDTH // 2, HEIGHT // 2)

TRACK_RX, TRACK_RY = 390, 270
TRACK_WIDTH = 110
INNER_RX = TRACK_RX - TRACK_WIDTH
INNER_RY = TRACK_RY - TRACK_WIDTH

SENSOR_ANGLES = (-70, -35, 0, 35, 70)
SENSOR_RANGE = 180
CAR_LENGTH, CAR_WIDTH = 26, 15

LAP_ANGLE = 2.0 * math.pi


def wrap_angle(angle):
    """Wrap radians to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def point_on_track(theta):
    """Return a point on the track centerline."""
    rx = TRACK_RX - TRACK_WIDTH / 2
    ry = TRACK_RY - TRACK_WIDTH / 2

    return (
        CENTER[0] + rx * math.cos(theta),
        CENTER[1] + ry * math.sin(theta),
    )


def is_on_track(x, y, margin=0.0):
    """Check whether a point is between the track edges."""
    dx = x - CENTER[0]
    dy = y - CENTER[1]

    outer_rx = TRACK_RX + margin
    outer_ry = TRACK_RY + margin
    inner_rx = INNER_RX - margin
    inner_ry = INNER_RY - margin

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


def track_theta(x, y):
    """Return the car's angular position around the elliptical track."""
    rx = TRACK_RX - TRACK_WIDTH / 2
    ry = TRACK_RY - TRACK_WIDTH / 2

    dx = (x - CENTER[0]) / rx
    dy = (y - CENTER[1]) / ry

    return math.atan2(dy, dx)


def track_tangent_angle(x, y):
    """Approximate the forward tangent of the ellipse."""
    theta = track_theta(x, y)

    rx = TRACK_RX - TRACK_WIDTH / 2
    ry = TRACK_RY - TRACK_WIDTH / 2

    tangent_x = -rx * math.sin(theta)
    tangent_y = ry * math.cos(theta)

    return math.atan2(tangent_y, tangent_x)


class Car:
    def __init__(
            self,
            x,
            y,
            angle,
            color=(40, 170, 255),
            brain=None,
    ):
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

        # Lap timing.
        self.completed_laps = 0
        self.best_lap_frames = None
        self.last_lap_frames = None
        self.lap_start_frame = 0
        self.lap_progress_angle = 0.0
        self.last_track_theta = track_theta(self.x, self.y)

    def get_inputs(self):
        """Return the seven inputs expected by the NEAT network."""
        self.sensors = [
            ray_distance(
                self.x,
                self.y,
                self.angle + math.radians(sensor_angle),
                )
            for sensor_angle in SENSOR_ANGLES
        ]

        sensor_inputs = [
            distance / SENSOR_RANGE
            for distance in self.sensors
        ]

        speed_input = max(
            0.0,
            min(1.0, self.speed / self.max_speed),
        )

        tangent = track_tangent_angle(self.x, self.y)
        direction_error = wrap_angle(tangent - self.angle)

        direction_input = max(
            -1.0,
            min(1.0, direction_error / math.pi),
        )

        return sensor_inputs + [speed_input, direction_input]

    def update_lap_progress(self):
        """Track forward progress and time completed laps."""
        current_theta = track_theta(self.x, self.y)

        delta = wrap_angle(current_theta - self.last_track_theta)
        self.last_track_theta = current_theta

        # Forward angular movement increases progress.
        # Reverse movement reduces incomplete-lap progress.
        self.lap_progress_angle = max(
            0.0,
            self.lap_progress_angle + delta,
            )

        while self.lap_progress_angle >= LAP_ANGLE:
            self.lap_progress_angle -= LAP_ANGLE

            self.completed_laps += 1

            lap_frames = self.frames_alive - self.lap_start_frame
            self.last_lap_frames = lap_frames
            self.lap_start_frame = self.frames_alive

            if lap_frames > 0:
                if (
                        self.best_lap_frames is None
                        or lap_frames < self.best_lap_frames
                ):
                    self.best_lap_frames = lap_frames

                # A completed lap receives a substantial bonus.
                # Faster laps receive a larger speed bonus.
                lap_seconds = lap_frames / 60.0

                self.fitness += 1000.0
                self.fitness += 3000.0 / max(lap_seconds, 0.5)

    def update(self):
        if not self.alive or self.brain is None:
            return

        outputs = self.brain.activate(self.get_inputs())

        steering = max(
            -1.0,
            min(1.0, float(outputs[0])),
        )

        throttle = max(
            -1.0,
            min(1.0, float(outputs[1])),
        )

        # Preserve the existing driving physics.
        self.angle += (
                steering
                * self.turn_rate
                * (0.45 + self.speed / self.max_speed)
        )

        if throttle >= 0:
            self.speed = min(
                self.max_speed,
                self.speed + self.acceleration * throttle,
                )
        else:
            self.speed = max(
                0.0,
                self.speed + self.brake * throttle,
                )

        old_x, old_y = self.x, self.y

        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed

        moved_distance = math.hypot(
            self.x - old_x,
            self.y - old_y,
            )

        self.distance_travelled += moved_distance
        self.frames_alive += 1

        # Crash penalty discourages reckless driving.
        if not is_on_track(self.x, self.y, margin=-3):
            self.alive = False
            self.fitness = max(0.0, self.fitness - 100.0)
            return

        tangent = track_tangent_angle(self.x, self.y)
        alignment = math.cos(
            wrap_angle(tangent - self.angle)
        )

        # Reward forward movement, with a small survival reward.
        forward_reward = max(0.0, self.speed * alignment)
        self.fitness += forward_reward + 0.03

        # Add lap completion and lap-time rewards.
        self.update_lap_progress()

    def draw(self, surface, show_sensors=False):
        if not self.alive:
            return

        body = pygame.Surface(
            (CAR_LENGTH, CAR_WIDTH),
            pygame.SRCALPHA,
        )

        pygame.draw.rect(
            body,
            self.color,
            (0, 0, CAR_LENGTH, CAR_WIDTH),
            border_radius=4,
        )

        pygame.draw.rect(
            body,
            (235, 240, 245),
            (
                CAR_LENGTH - 8,
                3,
                5,
                CAR_WIDTH - 6,
            ),
            border_radius=2,
        )

        rotated = pygame.transform.rotate(
            body,
            -math.degrees(self.angle),
        )

        rect = rotated.get_rect(
            center=(round(self.x), round(self.y))
        )

        surface.blit(rotated, rect)

        if show_sensors:
            for sensor_angle, distance in zip(
                    SENSOR_ANGLES,
                    self.sensors,
            ):
                end = (
                    self.x
                    + math.cos(
                        self.angle + math.radians(sensor_angle)
                    ) * distance,
                    self.y
                    + math.sin(
                        self.angle + math.radians(sensor_angle)
                    ) * distance,
                )

                pygame.draw.line(
                    surface,
                    (255, 220, 70),
                    (self.x, self.y),
                    end,
                    1,
                )
