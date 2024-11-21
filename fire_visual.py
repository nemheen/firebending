import cv2
import mediapipe as mp
import pygame
import random
import numpy as np
import pygame.display

# Initialize MediaPipe for hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

# Initialize OpenCV for webcam capture
cap = cv2.VideoCapture(0)

# Initialize Pygame
pygame.init()
width, height = 640, 480
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Firebender Effect")
clock = pygame.time.Clock()

def load_and_scale_image(path, scale_size):
    image = pygame.image.load(path)
    return pygame.transform.scale(image, scale_size)


target_size = (150, 150)
flames = [load_and_scale_image(f'images/flame_{i}_{j}.png', target_size) for i in range(3) for j in range(3)]


def blitter(x, y, frame_count):
    # Cycle through flame images to create animation
    flame_image = flames[frame_count % len(flames)]


    # scale_factor = 1 + 0.5 * np.sin(frame_count * 0.1)  # Adjust the multiplier and frequency as needed
    # scaled_flame = pygame.transform.scale(flame_image,
    #                                       (int(flame_image.get_width() * scale_factor),
    #                                        int(flame_image.get_height() * scale_factor)))

    # Center the scaled flame at the fingertip position
    flame_rect = flame_image.get_rect(center=(x, y))
    screen.blit(flame_image, flame_rect)

running = True
frame_count = 0

while running:
    # Capture frame from the webcam
    success, frame = cap.read()
    if not success:
        break

    # Flip and resize the frame for display
    # frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (width, height))

    # Convert frame to RGB to fix blue tint issue
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Display the camera feed as background
    frame_surface = pygame.surfarray.make_surface(np.rot90(rgb_frame))  # Rotate for Pygame orientation
    screen.blit(frame_surface, (0, 0))

    # Detect hand and track the fingertip for fire effect
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            x, y = int(index_finger_tip.x * width), int(index_finger_tip.y * height)

            # Display the flame following the fingertip
            blitter(x, y, frame_count)

    # Update the Pygame display
    pygame.display.flip()
    clock.tick(50)
    frame_count += 1  # Increase frame count for cycling flames

    # Exit on Pygame window close
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

# Release resources
cap.release()
pygame.quit()
