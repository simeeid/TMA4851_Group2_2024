#!/usr/bin/env python

import cv2
import dlib
import numpy as np

# Initialize the VideoCapture object to read from your camera.
cap = cv2.VideoCapture(0)

# Initialize dlib's face detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Screen dimensions (for demonstration, adjust as per your screen resolution)
screen_width = int(cap.get(3))
screen_height = int(cap.get(4))

# Define regions (three vertical and three horizontal lines)
vertical_lines = [screen_width // 3, 2 * screen_width // 3]
horizontal_lines = [screen_height // 3, 2 * screen_height // 3]

# Thresholds for moving to a new region
HORIZONTAL_THRESHOLD = 0.1
VERTICAL_THRESHOLD = 0.1

# Define regions (three vertical and three horizontal lines)
vertical_lines = [screen_width // 3, 2 * screen_width // 3]
horizontal_lines = [screen_height // 3, 2 * screen_height // 3]

def unit_vector(vector):
    """ Returns the unit vector of the vector. """
    return vector / np.linalg.norm(vector)

def draw_vector(frame, start_point, direction_vector, scale=5, color=(0, 255, 0), thickness=2):
    """Draws a vector (line) on the frame from a start point in a given direction, scaled for better visibility."""
    end_point = (int(start_point[0] + direction_vector[0] * scale), int(start_point[1] + direction_vector[1] * scale))
    cv2.line(frame, start_point, end_point, color, thickness)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray, 0)

    # Initialize circle_x and circle_y with default positions (center of the screen)
    circle_x, circle_y = screen_width // 2, screen_height // 2

    # Create an overlay for highlighting regions
    overlay = frame.copy()

    # Iterate over faces
    for face in faces:
        # Draw a rectangle around each face detected
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Get facial landmarks
        landmarks = predictor(gray, face)

        # Draw circles on facial landmarks
        for n in range(0, 68): # There are 68 landmark points
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        
        # Approximate "ear" points
        left_ear = np.array([landmarks.part(0).x, landmarks.part(0).y])
        right_ear = np.array([landmarks.part(16).x, landmarks.part(16).y])
        ear_distance = np.linalg.norm(right_ear - left_ear)
        
        # Midpoint between the "ear" points
        ear_midpoint = (left_ear + right_ear) / 2
        
        # Nose tip for forward direction
        nose_top = np.array([landmarks.part(27).x, landmarks.part(27).y])
        
        # Forward vector calculation
        #forward_vector = unit_vector(nose_top - ear_midpoint)
        forward_vector = nose_top - ear_midpoint

        # Calculate ratios
        horizontal_ratio = forward_vector[0] / ear_distance
        vertical_ratio = forward_vector[1] / ear_distance
        
        # Calculate circle position based on ratios (scaled for visibility)
        circle_x = int((horizontal_ratio / HORIZONTAL_THRESHOLD) * (screen_width // 6) + (screen_width // 2))
        circle_y = int((vertical_ratio / VERTICAL_THRESHOLD) * (screen_height // 6) + (screen_height // 2))

        # Ensure the circle position does not go beyond the screen boundaries
        circle_x = max(0, min(screen_width, circle_x))
        circle_y = max(0, min(screen_height, circle_y))

        # Determine the active region based on circle position
        region_x = 0 if circle_x < vertical_lines[0] else 1 if circle_x < vertical_lines[1] else 2
        region_y = 0 if circle_y < horizontal_lines[0] else 1 if circle_y < horizontal_lines[1] else 2

        # Highlight the active region
        cv2.rectangle(overlay, (region_x * screen_width // 3, region_y * screen_height // 3),
                      ((region_x + 1) * screen_width // 3, (region_y + 1) * screen_height // 3),
                      (0, 255, 0, 0.1), -1)  # Adjust color and transparency

        # Draw the forward vector from the nose tip
        draw_vector(frame, tuple(nose_top), forward_vector, scale=5, color=(255, 0, 0), thickness=2)

        # Determine orientation
        horizontal_orientation = "center"
        if horizontal_ratio > HORIZONTAL_THRESHOLD:
            horizontal_orientation = "right"
        elif horizontal_ratio < -HORIZONTAL_THRESHOLD:
            horizontal_orientation = "left"

        vertical_orientation = "center"
        if vertical_ratio > VERTICAL_THRESHOLD:
            vertical_orientation = "down"
        elif vertical_ratio < -VERTICAL_THRESHOLD:
            vertical_orientation = "up"

        orientation = f"{horizontal_orientation}-{vertical_orientation}"

        # Display the ratios and orientation indicator
        cv2.putText(frame, f"H: {horizontal_ratio:.2f}, V: {vertical_ratio:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, orientation, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Draw the vertical and horizontal lines
    for line in vertical_lines:
        cv2.line(frame, (line, 0), (line, screen_height), (255, 255, 255), 2)
    for line in horizontal_lines:
        cv2.line(frame, (0, line), (screen_width, line), (255, 255, 255), 2)

    # Draw the circle
    cv2.circle(frame, (circle_x, circle_y), 10, (0, 0, 255), -1)

    # Blend the overlay with the original frame
    alpha = 0.4  # Transparency factor
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Display the resulting frame with detections
    cv2.imshow('Frame', frame)

    # Break the loop with the escape key
    if cv2.waitKey(1) == 27:
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()