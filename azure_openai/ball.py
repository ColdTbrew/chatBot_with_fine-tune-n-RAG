import pygame
import math
import sys
import random

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
CENTER = (WIDTH // 2, HEIGHT // 2)
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Bouncing Balls in a Spinning Hexagon")
CLOCK = pygame.time.Clock()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)

# Physics constants
GRAVITY = 0.5  # pixels per frame^2
FRICTION = 0.99  # velocity damping
ELASTICITY = 0.9  # collision elasticity

# Hexagon properties
HEX_RADIUS = 200
NUM_SIDES = 6
angle_offset = 0  # initial rotation angle in degrees
rotation_speed = 50  # degrees per second

# Ball properties
BALL_RADIUS = [5, 10, 15, 20, 25]  # Different sizes for each ball
ball_colors = [RED, BLUE, GREEN, YELLOW, PURPLE]
balls = []

def get_hexagon_vertices(center, radius, angle_deg):
    """Calculate the vertices of a regular hexagon."""
    vertices = []
    angle_rad = math.radians(angle_deg)
    for i in range(NUM_SIDES):
        theta = angle_rad + i * (2 * math.pi / NUM_SIDES)
        x = center[0] + radius * math.cos(theta)
        y = center[1] + radius * math.sin(theta)
        vertices.append(pygame.math.Vector2(x, y))
    return vertices

def is_inside_hexagon(point, vertices):
    """Check if a point is inside a hexagon using ray-casting method."""
    count = 0
    for i in range(NUM_SIDES):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % NUM_SIDES]
        if ((v1.y > point.y) != (v2.y > point.y)) and \
                (point.x < (v2.x - v1.x) * (point.y - v1.y) / (v2.y - v1.y) + v1.x):
            count += 1
    return count % 2 == 1

def resolve_ball_collisions():
    """Prevent balls from overlapping by applying simple collision resolution."""
    for i in range(len(balls)):
        for j in range(i + 1, len(balls)):
            b1, b2 = balls[i], balls[j]
            dist = b1['pos'].distance_to(b2['pos'])
            min_dist = b1['radius'] + b2['radius']
            if dist < min_dist:
                overlap = min_dist - dist
                direction = (b1['pos'] - b2['pos']).normalize()
                b1['pos'] += direction * (overlap / 2)
                b2['pos'] -= direction * (overlap / 2)

# Initialize balls ensuring they are inside the hexagon and not overlapping
for i in range(len(BALL_RADIUS)):
    while True:
        position = pygame.math.Vector2(
            random.uniform(CENTER[0] - HEX_RADIUS, CENTER[0] + HEX_RADIUS),
            random.uniform(CENTER[1] - HEX_RADIUS, CENTER[1] + HEX_RADIUS)
        )
        if is_inside_hexagon(position, get_hexagon_vertices(CENTER, HEX_RADIUS, angle_offset)):
            valid = all(position.distance_to(b['pos']) > (b['radius'] + BALL_RADIUS[i]) for b in balls)
            if valid:
                break
    velocity = pygame.math.Vector2(random.uniform(-5, 5), random.uniform(-5, 5))
    balls.append({'pos': position, 'vel': velocity, 'radius': BALL_RADIUS[i], 'color': ball_colors[i]})

def main():
    global angle_offset
    running = True
    angular_velocity = math.radians(rotation_speed)

    while running:
        dt = CLOCK.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        angle_offset += math.degrees(angular_velocity * dt)
        angle_offset %= 360

        vertices = get_hexagon_vertices(CENTER, HEX_RADIUS, angle_offset)

        for ball in balls:
            ball['vel'].y += GRAVITY
            ball['vel'] *= FRICTION
            ball['pos'] += ball['vel'] * dt
            if not is_inside_hexagon(ball['pos'], vertices):
                ball['vel'] *= -ELASTICITY

        resolve_ball_collisions()
        SCREEN.fill(BLACK)
        pygame.draw.polygon(SCREEN, WHITE, vertices, 2)
        for ball in balls:
            pygame.draw.circle(SCREEN, ball['color'], (int(ball['pos'].x), int(ball['pos'].y)), ball['radius'])
        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
