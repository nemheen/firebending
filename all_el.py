import cv2
import mediapipe as mp
import pygame
import random
import numpy as np

# Initialize MediaPipe for hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

# Initialize OpenCV for webcam capture
cap = cv2.VideoCapture(0)

# Initialize Pygame
pygame.init()
width, height = 640, 480
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Element Bending Effect")
clock = pygame.time.Clock()

# Particle properties for different elements
particles = []


# suurig n 3jindu, deegure flow like
def add_fire_particles(x, y):
    for _ in range(10):
        angle = random.uniform(-0.5, 0.5)
        speed = random.uniform(3, 5)
        particles.append([[x, y], [speed * np.cos(angle), -speed * np.sin(angle)], random.randint(5, 10), (255, random.randint(100, 200), 0), 'fire'])


# smooth water drop like features
def add_water_particles(x, y):
    for _ in range(8):
        speed = random.uniform(1, 2)
        particles.append([[x, y], [random.uniform(-1, 1), speed], random.randint(5, 8), (0, random.randint(100, 200), 255), 'water'])

# 4jin helber
def add_rock_particles(x, y):
    for _ in range(6):
        speed = random.uniform(1, 3)
        particles.append([[x, y], [random.uniform(-1, 1), speed], random.randint(8, 12), (random.randint(100, 120), 100, 60), 'rock'])

#keep it same
def add_air_particles(x, y):
    for _ in range(15):
        angle = random.uniform(0, 2 * np.pi)
        speed = random.uniform(0.5, 1.5)
        particles.append([[x, y], [speed * np.cos(angle), speed * np.sin(angle)], random.randint(3, 6), (200, 200, 200), 'air'])

def update_particles():
    for particle in particles[:]:
        # Apply velocity to position
        particle[0][0] += particle[1][0]
        particle[0][1] += particle[1][1]

        # Particle behavior based on type
        if particle[4] == 'fire':
            particle[1][1] += 0.05  # Upward force for fire
            particle[1][0] *= 0.98
        elif particle[4] == 'water':
            particle[1][1] += 0.1  # Gravity effect for water
        elif particle[4] == 'rock':
            particle[1][1] += 0.2  # Stronger gravity for rock
            if particle[0][1] >= height - 20:  # Simulate ground collision
                particle[1][1] *= -0.5  # Bounce up with reduced speed
        elif particle[4] == 'air':
            particle[1][0] += random.uniform(-0.1, 0.1)  # Random drift for air

        # Decrease particle size for fade-out effect
        particle[2] -= 0.1
        if particle[2] <= 0:
            particles.remove(particle)

def draw_particles():
    for particle in particles:
        pygame.draw.circle(screen, particle[3], [int(particle[0][0]), int(particle[0][1])], int(particle[2]))

running = True

while running:
    # Capture frame from the webcam
    success, frame = cap.read()
    if not success:
        break

    # Flip and resize the frame for display
    # frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (width, height))

    # Convert frame to RGB and process with MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Display the camera feed as background
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.rot90(frame)  # Rotate frame for Pygame (optional based on orientation)
    frame_surface = pygame.surfarray.make_surface(frame)
    screen.blit(frame_surface, (0, 0))

    # Detect hands and assign each element to a hand
    if result.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            # Alternate between elements based on hand index
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(index_finger_tip.x * width), int(index_finger_tip.y * height)

            if idx == 0:  # First hand controls fire and water
                add_fire_particles(x, y)
                add_water_particles(x, y + 50)  # Offset water particles slightly below fire
            elif idx == 1:  # Second hand controls rock and air
                add_rock_particles(x, y)
                add_air_particles(x, y - 50)  # Offset air particles slightly above rock

    # Update and draw particles
    update_particles()
    draw_particles()

    # Update the Pygame display
    pygame.display.flip()
    clock.tick(30)

    # Exit on Pygame window close
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

# Release resources
cap.release()
pygame.quit()
