import cv2
import mediapipe as mp
import numpy as np


def add_fire_effect(frame, x, y):
    # Load or generate a small fire image
    fire_img = cv2.imread("images/fiire.png", cv2.IMREAD_UNCHANGED)  # Use a transparent fire image
    fire_img = cv2.resize(fire_img, (50, 50))  # Resize fire effect
    if fire_img is None:
        print("Error: Could not load fire image. Check the file path.")
        exit()
    # Ensure fire effect fits within the frame
    y1, y2 = max(0, y - 25), min(frame.shape[0], y + 25)
    x1, x2 = max(0, x - 25), min(frame.shape[1], x + 25)
    alpha_fire = fire_img[:, :, 3] / 255.0
    for c in range(3):
        frame[y1:y2, x1:x2, c] = (1 - alpha_fire) * frame[y1:y2, x1:x2, c] + alpha_fire * fire_img[:, :, c]
    return frame


# Initialize MediaPipe hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip the frame for natural mirroring
    frame = cv2.flip(frame, 1)

    # Convert frame to RGB as MediaPipe uses RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks for creating a fire effect
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            x, y = int(index_finger_tip.x * frame.shape[1]), int(index_finger_tip.y * frame.shape[0])
            print(frame.shape)
            # Call the function to draw fire effect at the fingertip position
            frame = add_fire_effect(frame, x, y)

    # Display the frame
    cv2.imshow('Firebender Effect', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
