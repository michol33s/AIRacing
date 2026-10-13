
import math
import pygame

WIDTH, HEIGHT = 1000, 800
BG = GRASS = (30, 130, 50)
TRACK_COLOR = (65, 65, 65)
ROAD_LINE = (240, 240, 240)
CENTER_LINE = (255, 220, 80)
TURQUOISE = (0, 255, 210)
TRACK_WIDTH, ROAD_MARGIN, SENSOR_RANGE = 100, 8, 150

TRACK_POINTS = [
    (180,650), (180,450), (220,300), (350,220),
    (520,200), (700,230), (820,330), (830,430),
    (780,490), (680,480), (620,410), (580,330),
    (500,300), (400,330), (350,430), (350,570),
    (300,650)
]


def build_track():
    points, n = [], len(TRACK_POINTS)
    for i in range(n):
        p0, p1 = TRACK_POINTS[(i-1)%n], TRACK_POINTS[i]
        p2, p3 = TRACK_POINTS[(i+1)%n], TRACK_POINTS[(i+2)%n]
        for j in range(12):
            t = j/12
            t2, t3 = t*t, t*t*t
            x = .5*(2*p1[0]+(-p0[0]+p2[0])*t+
                    (2*p0[0]-5*p1[0]+4*p2[0]-p3[0])*t2+
                    (-p0[0]+3*p1[0]-3*p2[0]+p3[0])*t3)
            y = .5*(2*p1[1]+(-p0[1]+p2[1])*t+
                    (2*p0[1]-5*p1[1]+4*p2[1]-p3[1])*t2+
                    (-p0[1]+3*p1[1]-3*p2[1]+p3[1])*t3)
            points.append((x,y))
    return points


TRACK_CENTERLINE = build_track()


def closest_track_index(x, y):
    return min(range(len(TRACK_CENTERLINE)),
               key=lambda i: (TRACK_CENTERLINE[i][0]-x)**2 +
                             (TRACK_CENTERLINE[i][1]-y)**2)


def closest_track_point(x, y):
    return TRACK_CENTERLINE[closest_track_index(x,y)]


def point_on_road(x, y):
    p = closest_track_point(x,y)
    return math.hypot(x-p[0], y-p[1]) < TRACK_WIDTH/2-ROAD_MARGIN


class Car:
    def __init__(self, theta=0):
        n = len(TRACK_CENTERLINE)
        i = int(theta*n/(2*math.pi)) % n
        self.x, self.y = TRACK_CENTERLINE[i]
        self.angle = self.track_tangent()
        self.speed, self.max_speed = 4.0, 12.0
        self.alive, self.fitness = True, 0.0
        self.sensor_readings, self.sensor_endpoints = [], []
        self.completed_laps, self.best_lap_frames = 0, None
        self.frames = self.lap_frames = self.stuck_frames = 0
        self.last_progress = i

    def track_tangent(self):
        i = closest_track_index(self.x, self.y)
        a = TRACK_CENTERLINE[i]
        b = TRACK_CENTERLINE[(i+1)%len(TRACK_CENTERLINE)]
        return math.atan2(b[1]-a[1], b[0]-a[0])

    def sensors(self):
        readings, endpoints = [], []
        for offset in (-math.pi/2, -math.pi/4, 0, math.pi/4, math.pi/2):
            a = self.angle+offset
            end = (self.x+math.cos(a)*SENSOR_RANGE,
                   self.y+math.sin(a)*SENSOR_RANGE)
            hit = SENSOR_RANGE
            for d in range(5, SENSOR_RANGE+1, 5):
                px, py = self.x+math.cos(a)*d, self.y+math.sin(a)*d
                if not point_on_road(px,py):
                    hit, end = d, (px,py)
                    break
            readings.append(hit/SENSOR_RANGE)
            endpoints.append(end)
        self.sensor_readings, self.sensor_endpoints = readings, endpoints
        return readings

    def get_sensors(self):
        return self.sensors()

    def update(self, outputs):
        if not self.alive:
            return

        steering = max(-1, min(1, outputs[0]))
        throttle = max(0, min(1, outputs[1]))
        self.angle += steering*0.055*max(0.5, self.speed/3)
        self.speed = min(self.max_speed, max(0, self.speed+throttle*0.12-0.025))
        self.x += math.cos(self.angle)*self.speed
        self.y += math.sin(self.angle)*self.speed
        self.frames += 1
        self.lap_frames += 1

        previous = self.last_progress
        i = closest_track_index(self.x, self.y)
        n = len(TRACK_CENTERLINE)
        delta = (i-previous+n//2) % n-n//2

        # Reward forward progress and speed.
        self.fitness += delta*0.5 + self.speed*0.01

        # Reward completing a lap in the forward direction.
        if previous > n*0.90 and i < n*0.10 and delta > 0:
            self.completed_laps += 1
            self.fitness += 100
            if self.best_lap_frames is None or self.lap_frames < self.best_lap_frames:
                self.best_lap_frames = self.lap_frames
            self.lap_frames = 0

        self.last_progress = i
        self.stuck_frames = self.stuck_frames+1 if self.speed < 0.3 else 0

        if self.stuck_frames > 240 or not point_on_road(self.x,self.y):
            self.alive = False
            self.fitness -= 5


    def draw(self, screen, show_sensors=False):
        if show_sensors:
            self.sensors()
            for e in self.sensor_endpoints:
                pygame.draw.line(screen, (255,220,0),
                                 (int(self.x),int(self.y)), (int(e[0]),int(e[1])), 1)

        s = pygame.Surface((64,30), pygame.SRCALPHA)
        dark, tyre = (25,29,35), (12,13,16)

        # Slick tyres
        for x,y in [(13,1),(13,22),(44,1),(44,22)]:
            pygame.draw.rect(s,tyre,(x,y,10,7),border_radius=2)

        # Front and rear wings
        pygame.draw.polygon(s,dark,[(48,2),(63,4),(63,7),(48,9),(45,7)])
        pygame.draw.polygon(s,dark,[(48,28),(63,26),(63,23),(48,21),(45,23)])
        pygame.draw.rect(s,dark,(3,1,12,28),border_radius=2)

        # Chassis and sidepods
        pygame.draw.polygon(s,TURQUOISE,
                            [(8,11),(20,8),(35,10),(54,13),(54,17),(35,20),(20,22),(8,19)])
        pygame.draw.polygon(s,(0,175,155),
                            [(21,8),(36,10),(42,13),(31,14),(22,12)])
        pygame.draw.polygon(s,(0,175,155),
                            [(21,22),(36,20),(42,17),(31,16),(22,18)])

        # Cockpit and halo
        pygame.draw.ellipse(s,dark,(29,11,13,8))
        pygame.draw.ellipse(s,(70,80,90),(32,12,7,6))
        pygame.draw.arc(s,(235,242,245),(28,10,15,10),0,math.tau,2)

        r = pygame.transform.rotate(s,-math.degrees(self.angle))
        screen.blit(r,r.get_rect(center=(int(self.x),int(self.y))))
