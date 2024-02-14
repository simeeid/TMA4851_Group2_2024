#!/usr/bin/env python

import cv2
import dlib
import numpy as np

class FaceDirectionTracker:
    # Thresholds for determining face orientation
    HORIZONTAL_THRESHOLD = 0.1
    VERTICAL_THRESHOLD = 0.1

    def __init__(self):
        # Initialize video capture, face detector, and shape predictor
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        
        # Get screen dimensions
        self.screen_width = int(self.cap.get(3))
        self.screen_height = int(self.cap.get(4))
        
        # Define lines for dividing the screen into regions
        self.vertical_lines = [self.screen_width // 3, 2 * self.screen_width // 3]
        self.horizontal_lines = [self.screen_height // 3, 2 * self.screen_height // 3]

    def draw_vector(self, frame, start_point, direction_vector, scale=5, color=(0, 255, 0), thickness=2):
        # Draw a vector (line) on the frame to indicate direction
        end_point = (int(start_point[0] + direction_vector[0] * scale), int(start_point[1] + direction_vector[1] * scale))
        cv2.line(frame, start_point, end_point, color, thickness)

    def draw_lines(self, frame):
        # Draw vertical and horizontal lines on the frame to divide it into regions
        for line in self.vertical_lines:
            cv2.line(frame, (line, 0), (line, self.screen_height), (255, 255, 255), 2)
        for line in self.horizontal_lines:
            cv2.line(frame, (0, line), (self.screen_width, line), (255, 255, 255), 2)

    def track(self):
        # Main tracking loop
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray, 0)
            overlay = frame.copy()

            for face in faces:
                self.process_face(frame, gray, face, overlay)

            self.draw_lines(frame)
            alpha = 0.4  # Transparency factor for overlay
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            cv2.imshow('Frame', frame)

            if cv2.waitKey(1) == 27:  # Exit on ESC
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def process_face(self, frame, gray, face, overlay):
        # Process each detected face
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        landmarks = self.predictor(gray, face)

        # Draw facial landmarks
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # Calculate forward vector from ear to nose top
        left_ear = np.array([landmarks.part(0).x, landmarks.part(0).y])
        right_ear = np.array([landmarks.part(16).x, landmarks.part(16).y])
        ear_distance = np.linalg.norm(right_ear - left_ear)
        ear_midpoint = (left_ear + right_ear) / 2
        nose_top = np.array([landmarks.part(27).x, landmarks.part(27).y])
        forward_vector = nose_top - ear_midpoint

        # Calculate orientation ratios
        horizontal_ratio = forward_vector[0] / ear_distance
        vertical_ratio = forward_vector[1] / ear_distance

        # Determine the region of interest based on orientation
        circle_x = int((horizontal_ratio / self.HORIZONTAL_THRESHOLD) * (self.screen_width // 6) + (self.screen_width // 2))
        circle_y = int((vertical_ratio / self.VERTICAL_THRESHOLD) * (self.screen_height // 6) + (self.screen_height // 2))
        circle_x = max(0, min(self.screen_width, circle_x))
        circle_y = max(0, min(self.screen_height, circle_y))

        # Determine which region the face is pointing towards
        region_x = 0 if circle_x < self.vertical_lines[0] else 1 if circle_x < self.vertical_lines[1] else 2
        region_y = 0 if circle_y < self.horizontal_lines[0] else 1 if circle_y < self.horizontal_lines[1] else 2

        # Highlight the region in the overlay
        cv2.rectangle(overlay, (region_x * self.screen_width // 3, region_y * self.screen_height // 3),
                      ((region_x + 1) * self.screen_width // 3, (region_y + 1) * self.screen_height // 3),
                      (0, 255, 0, 0.1), -1)

        # Draw the forward vector on the frame
        self.draw_vector(frame, tuple(nose_top), forward_vector, scale=5, color=(255, 0, 0), thickness=2)

        # Determine the horizontal and vertical orientation of the face
        horizontal_orientation = "center"
        if horizontal_ratio > self.HORIZONTAL_THRESHOLD:
            horizontal_orientation = "right"
        elif horizontal_ratio < -self.HORIZONTAL_THRESHOLD:
            horizontal_orientation = "left"

        vertical_orientation = "center"
        if vertical_ratio > self.VERTICAL_THRESHOLD:
            vertical_orientation = "down"
        elif vertical_ratio < -self.VERTICAL_THRESHOLD:
            vertical_orientation = "up"

        # Combine the orientation information
        orientation = f"{horizontal_orientation}-{vertical_orientation}"

        # Display the orientation and ratio information on the frame
        cv2.putText(frame, f"H: {horizontal_ratio:.2f}, V: {vertical_ratio:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, orientation, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw a circle in the region the face is pointing towards
        cv2.circle(frame, (circle_x, circle_y), 10, (0, 0, 255), -1)

if __name__ == "__main__":
    tracker = FaceDirectionTracker()
    tracker.track()