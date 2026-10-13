
import math
import random
import pygame

# --------------------------------------------------
# DISPLAY AND TRACK SETTINGS
# --------------------------------------------------

WIDTH, HEIGHT = 1200, 800
CENTER = (WIDTH // 2, HEIGHT // 2)

TRACK_RX = 390
TRACK_RY = 270

TRACK_WIDTH = 110

INNER_RX = TRACK_RX - TRACK_WIDTH
INNER_RY = TRACK_RY - TRACK_WIDTH

BG = (12, 16, 25)
TRACK_COLOR = (48, 54, 65)
ROAD_LINE = (220, 225, 235)
TURQUOISE = (40, 230, 210)

WHITE = (245, 245, 245)
BLACK = (15, 15, 18)
RED = (235, 45, 55)
DARK_RED = (150, 20, 30)
GOLD = (255, 195, 55)
WINDOW_BLUE = (65, 150, 195)

# --------------------------------------------------
# CAR SETTINGS
# --------------------------------------------------

CAR_LENGTH = 30
CAR_WIDTH = 15

SENSOR_ANGLES = (-70, -35, 0, 35, 70)
SENSOR_RANGE = 180

MAX_SPEED = 5.0
ACCELERATION = 0.12
BRAKING = 0.08
FRICTION = 0.025
STEERING_RATE = 0.055

ROAD_MARGIN = 7





class Car:
    def __init__(self, theta=None):
        """
        Create a car at a random position around the circuit,
        or at a supplied theta angle in radians.
        """

        self.theta = (
            random.uniform(0, 2 * math.pi)
            if theta is None
            else theta
        )

        # Place the car approximately in the middle of the road.
        road_rx = (TRACK_RX + INNER_RX) / 2
        road_ry = (TRACK_RY + INNER_RY) / 2

        self.x = CENTER[0] + road_rx * math.cos(self.theta)
        self.y = CENTER[1] + road_ry * math.sin(self.theta)

        # Align the car with the direction of travel around the oval.
        self.angle = self.track_tangent()

        self.speed = 0.0
        self.max_speed = MAX_SPEED
        self.alive = True

        self.fitness = 0.0
        self.distance_reward = 0.0

        self.completed_laps = 0
        self.best_lap_frames = None
        self.lap_frames = 0

        self.previous_progress = self.track_progress()
        self.progress_since_lap = 0.0

        self.stuck_frames = 0
        self.previous_x = self.x
        self.previous_y = self.y

        self.color = (
            random.randint(100, 255),
            random.randint(60, 220),
            random.randint(60, 220),
        )

    # --------------------------------------------------
    # TRACK GEOMETRY
    # --------------------------------------------------

    def track_progress(self):
        """
        Return the car's angular position around the ellipse.
        """

        dx = (self.x - CENTER[0]) / TRACK_RX
        dy = (self.y - CENTER[1]) / TRACK_RY

        return math.atan2(dy, dx) % (2 * math.pi)

    def track_tangent(self):
        """
        Calculate the direction tangent to the oval at the car's
        current position.
        """

        theta = math.atan2(
            (self.y - CENTER[1]) / TRACK_RY,
            (self.x - CENTER[0]) / TRACK_RX,
            )

        dx = -TRACK_RX * math.sin(theta)
        dy = TRACK_RY * math.cos(theta)

        return math.atan2(dy, dx)

    def point_on_road(self, x, y):
        """
        Check whether a point lies between the track boundaries.
        """

        outer_rx = TRACK_RX - ROAD_MARGIN
        outer_ry = TRACK_RY - ROAD_MARGIN

        inner_rx = INNER_RX + ROAD_MARGIN
        inner_ry = INNER_RY + ROAD_MARGIN

        ox = (x - CENTER[0]) / outer_rx
        oy = (y - CENTER[1]) / outer_ry

        ix = (x - CENTER[0]) / inner_rx
        iy = (y - CENTER[1]) / inner_ry

        inside_outer = ox * ox + oy * oy <= 1.0
        outside_inner = ix * ix + iy * iy >= 1.0

        return inside_outer and outside_inner

    def on_road(self):
        """
        Check the center and corners of the car against the track.
        """

        forward_x = math.cos(self.angle)
        forward_y = math.sin(self.angle)

        side_x = -math.sin(self.angle)
        side_y = math.cos(self.angle)

        front = CAR_LENGTH * 0.42
        rear = CAR_LENGTH * 0.42
        side = CAR_WIDTH * 0.42

        points = [
            (self.x, self.y),
            (
                self.x + forward_x * front + side_x * side,
                self.y + forward_y * front + side_y * side,
            ),
            (
                self.x + forward_x * front - side_x * side,
                self.y + forward_y * front - side_y * side,
            ),
            (
                self.x - forward_x * rear + side_x * side,
                self.y - forward_y * rear + side_y * side,
            ),
            (
                self.x - forward_x * rear - side_x * side,
                self.y - forward_y * rear - side_y * side,
            ),
        ]

        return all(self.point_on_road(px, py) for px, py in points)

    # --------------------------------------------------
    # SENSORS
    # --------------------------------------------------

    def cast_sensor(self, angle):
        """
        Cast a ray from the car until it reaches the track edge.
        Return a normalized distance from 0 to 1.
        """

        for distance in range(0, SENSOR_RANGE + 1, 4):
            px = self.x + math.cos(angle) * distance
            py = self.y + math.sin(angle) * distance

            if not self.point_on_road(px, py):
                return distance / SENSOR_RANGE

        return 1.0

    def sensors(self):
        """
        Five forward-facing sensors, normalized from 0 to 1.
        """

        readings = []

        for degrees in SENSOR_ANGLES:
            sensor_angle = self.angle + math.radians(degrees)
            readings.append(self.cast_sensor(sensor_angle))

        return readings

    def sensor_endpoints(self):
        endpoints = []

        for degrees in SENSOR_ANGLES:
            sensor_angle = self.angle + math.radians(degrees)

            distance = self.cast_sensor(sensor_angle) * SENSOR_RANGE

            end_x = self.x + math.cos(sensor_angle) * distance
            end_y = self.y + math.sin(sensor_angle) * distance

            endpoints.append((end_x, end_y))

        return endpoints

    # --------------------------------------------------
    # LAP PROGRESS AND FITNESS
    # --------------------------------------------------

    def update_lap_progress(self):
        """
        Track progress around the circuit and detect completed laps.
        """

        current_progress = self.track_progress()

        delta = current_progress - self.previous_progress

        # Handle crossing the 0 / 2*pi boundary.
        if delta < -math.pi:
            delta += 2 * math.pi
        elif delta > math.pi:
            delta -= 2 * math.pi

        # Only reward forward progress.
        if delta > 0:
            self.progress_since_lap += delta
            self.fitness += delta * 10.0

        self.previous_progress = current_progress

        # A complete revolution is approximately 2*pi radians.
        if self.progress_since_lap >= 2 * math.pi:
            self.progress_since_lap -= 2 * math.pi
            self.completed_laps += 1

            if (
                    self.best_lap_frames is None
                    or self.lap_frames < self.best_lap_frames
            ):
                self.best_lap_frames = self.lap_frames

            self.lap_frames = 0

            # Reward completing a full lap.
            self.fitness += 500.0

    # --------------------------------------------------
    # UPDATE
    # --------------------------------------------------

    def update(self, outputs):
        """
        Update steering and throttle from the NEAT network.

        Expected outputs:
            outputs[0] = steering
            outputs[1] = throttle
        """

        if not self.alive:
            return

        steering = max(-1.0, min(1.0, outputs[0]))
        throttle = max(-1.0, min(1.0, outputs[1]))

        # Steer more effectively as the car gains speed.
        steering_strength = 0.45 + 0.55 * (
            min(abs(self.speed) / self.max_speed, 1.0)
        )

        self.angle += (
                steering
                * STEERING_RATE
                * steering_strength
        )

        # Throttle and braking.
        if throttle >= 0:
            self.speed += throttle * ACCELERATION
        else:
            self.speed += throttle * BRAKING

        # Natural drag.
        if self.speed > 0:
            self.speed = max(0.0, self.speed - FRICTION)
        elif self.speed < 0:
            self.speed = min(0.0, self.speed + FRICTION)

        self.speed = max(0.0, min(self.max_speed, self.speed))

        # Move in the direction the car is facing.
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed

        self.lap_frames += 1

        # Detect whether the car has left the road.
        if not self.on_road():
            self.alive = False
            self.fitness -= 25.0
            return

        self.update_lap_progress()

        # Penalize remaining stationary for too long.
        movement = math.hypot(
            self.x - self.previous_x,
            self.y - self.previous_y,
            )

        if movement < 0.2:
            self.stuck_frames += 1
        else:
            self.stuck_frames = 0

        self.previous_x = self.x
        self.previous_y = self.y

        if self.stuck_frames > 180:
            self.alive = False
            self.fitness -= 10.0
            return

        # Small reward for staying alive and moving.
        self.fitness += 0.01

    # --------------------------------------------------
    # DRAWING
    # --------------------------------------------------

    def draw(self, screen, show_sensors=False):
        """
        Draw the car, tires, windows, and optional sensor rays.
        """

        if show_sensors:
            for endpoint in self.sensor_endpoints():
                pygame.draw.line(
                    screen,
                    (70, 220, 130),
                    (int(self.x), int(self.y)),
                    (int(endpoint[0]), int(endpoint[1])),
                    1,
                )

        # Local car coordinates: forward is positive X.
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)

        def world_point(local_x, local_y):
            return (
                int(self.x + local_x * cos_a - local_y * sin_a),
                int(self.y + local_x * sin_a + local_y * cos_a),
            )

        # Tire positions in the car's local coordinate system.
        tire_width = 9
        tire_height = 4

        tire_positions = [
            (7, -CAR_WIDTH / 2 - 1),
            (7, CAR_WIDTH / 2 + 1),
            (-7, -CAR_WIDTH / 2 - 1),
            (-7, CAR_WIDTH / 2 + 1),
        ]

        # IMPORTANT: rotate tire surfaces, never Rect objects.
        for local_x, local_y in tire_positions:
            tire_surface = pygame.Surface(
                (tire_width, tire_height),
                pygame.SRCALPHA,
            )

            pygame.draw.rect(
                tire_surface,
                BLACK,
                (0, 0, tire_width, tire_height),
                border_radius=1,
            )

            rotated_tire = pygame.transform.rotate(
                tire_surface,
                -math.degrees(self.angle),
            )

            tire_rect = rotated_tire.get_rect(
                center=world_point(local_x, local_y)
            )

            screen.blit(rotated_tire, tire_rect)

        # Main body.
        body_local = [
            (CAR_LENGTH / 2, 0),
            (CAR_LENGTH / 2 - 5, -CAR_WIDTH / 2),
            (-CAR_LENGTH / 2 + 3, -CAR_WIDTH / 2),
            (-CAR_LENGTH / 2, -CAR_WIDTH / 2 + 3),
            (-CAR_LENGTH / 2, CAR_WIDTH / 2 - 3),
            (-CAR_LENGTH / 2 + 3, CAR_WIDTH / 2),
            (CAR_LENGTH / 2 - 5, CAR_WIDTH / 2),
        ]

        body_points = [
            world_point(px, py) for px, py in body_local
        ]

        pygame.draw.polygon(screen, DARK_RED, body_points)

        inset_local = [
            (CAR_LENGTH / 2 - 2, 0),
            (CAR_LENGTH / 2 - 6, -CAR_WIDTH / 2 + 2),
            (-CAR_LENGTH / 2 + 4, -CAR_WIDTH / 2 + 2),
            (-CAR_LENGTH / 2 + 1, 0),
            (-CAR_LENGTH / 2 + 4, CAR_WIDTH / 2 - 2),
            (CAR_LENGTH / 2 - 6, CAR_WIDTH / 2 - 2),
        ]

        inset_points = [
            world_point(px, py) for px, py in inset_local
        ]

        pygame.draw.polygon(screen, self.color, inset_points)

        # Cockpit / windshield.
        cockpit_local = [
            (6, -CAR_WIDTH * 0.29),
            (1, -CAR_WIDTH * 0.29),
            (-5, -CAR_WIDTH * 0.24),
            (-7, 0),
            (-5, CAR_WIDTH * 0.24),
            (1, CAR_WIDTH * 0.29),
            (6, CAR_WIDTH * 0.29),
        ]

        cockpit_points = [
            world_point(px, py) for px, py in cockpit_local
        ]

        pygame.draw.polygon(
            screen,
            WINDOW_BLUE,
            cockpit_points,
        )

        # Nose highlight.
        pygame.draw.line(
            screen,
            GOLD,
            world_point(CAR_LENGTH / 2 - 3, 0),
            world_point(CAR_LENGTH / 2 - 8, 0),
            2,
        )

        # Center marker.
        pygame.draw.circle(
            screen,
            WHITE,
            (int(self.x), int(self.y)),
            2,
        )
