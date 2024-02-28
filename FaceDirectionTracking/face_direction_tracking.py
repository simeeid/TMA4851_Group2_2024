#!/usr/bin/env python

import cv2
import dlib
import numpy as np

class FaceDirectionTracker:
    # Thresholds for determining face orientation
    HORIZONTAL_THRESHOLD = 15
    VERTICAL_THRESHOLD = 15

    def __init__(self):
        # Initialize video capture, face detector, and shape predictor
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        
        # Calculation method
        self.method = 'improved'

        # Get screen dimensions
        self.screen_width = int(self.cap.get(3))
        self.screen_height = int(self.cap.get(4))
        
        # Define lines for dividing the screen into regions
        self.vertical_lines = [self.screen_width // 3, 2 * self.screen_width // 3]
        self.horizontal_lines = [self.screen_height // 3, 2 * self.screen_height // 3]

        # Camera internals
        self.focal_length = self.screen_width
        self.center = (self.screen_width / 2, self.screen_height / 2)
        self.camera_matrix = np.array(
            [[self.focal_length, 0, self.center[0]],
             [0, self.focal_length, self.center[1]],
             [0, 0, 1]], dtype="double"
        )
        self.dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

        # 3D model points.
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])

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


    # Method 1: Using pose estimation
    def calculate_direction_simple(self, landmarks, overlay):
        SENSITIVITY = 100
        # Calculate forward vector from ear to nose top
        left_ear = np.array([landmarks.part(0).x, landmarks.part(0).y])
        right_ear = np.array([landmarks.part(16).x, landmarks.part(16).y])
        ear_distance = np.linalg.norm(right_ear - left_ear)
        ear_midpoint = (left_ear + right_ear) / 2
        nose_top = np.array([landmarks.part(27).x, landmarks.part(27).y])
        forward_vector = nose_top - ear_midpoint

        # Calculate orientation ratios
        horizontal_angle = - SENSITIVITY * forward_vector[0] / ear_distance
        vertical_angle = - SENSITIVITY * forward_vector[1] / ear_distance

        # Draw the forward vector on the frame
        self.draw_vector(overlay, tuple(nose_top), forward_vector, scale=5, color=(255, 0, 0), thickness=2)

        return horizontal_angle, vertical_angle

    # Method 2: Using geometric approach
    def calculate_direction_improved(self, landmarks, overlay):
        # 2D image points. If you change the image, you need to change vector
        image_points = np.array([
            (landmarks.part(30).x, landmarks.part(30).y), # Nose tip
            (landmarks.part(8).x, landmarks.part(8).y),   # Chin
            (landmarks.part(36).x, landmarks.part(36).y), # Left eye left corner
            (landmarks.part(45).x, landmarks.part(45).y), # Right eye right corner
            (landmarks.part(48).x, landmarks.part(48).y), # Left Mouth corner
            (landmarks.part(54).x, landmarks.part(54).y)  # Right Mouth corner
        ], dtype="double")

        # Solve for pose
        _, rotation_vector, translation_vector = cv2.solvePnP(self.model_points, image_points, self.camera_matrix, self.dist_coeffs)

        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # Construct a 3x4 projection matrix from the rotation matrix and translation vector
        projection_matrix = np.hstack((rotation_matrix, translation_vector))

        # Now you can safely decompose the projection matrix
        euler_angles = cv2.decomposeProjectionMatrix(projection_matrix)[-1]

        # Calculate horizontal and vertical angles
        horizontal_angle = euler_angles[1][0]  # Horizontal rotation
        vertical_angle = 180 - (euler_angles[0][0] % 360) # Vertical rotation

        # Project a 3D point to draw the direction vector
        nose_end_point3D = np.array([(0.0, 0.0, 1000.0)])
        nose_end_point2D, _ = cv2.projectPoints(nose_end_point3D, rotation_vector, translation_vector, self.camera_matrix, self.dist_coeffs)

        # Use draw_vector to draw the line
        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        direction_vector = (p2[0] - p1[0], p2[1] - p1[1])
        self.draw_vector(overlay, p1, direction_vector, scale=0.4, color=(255, 0, 0), thickness=2)

        return horizontal_angle, vertical_angle

    def process_face(self, frame, gray, face, overlay):
        # Process each detected face
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        landmarks = self.predictor(gray, face)

        # Choose the calculation method
        if self.method == 'simple':
            # Pose estimation calculations
            horizontal_angle, vertical_angle = self.calculate_direction_simple(landmarks, frame)
        elif self.method == 'improved':
            # Geometric calculations
            horizontal_angle, vertical_angle = self.calculate_direction_improved(landmarks, overlay)
        # Draw facial landmarks
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # Determine the region of interest based on orientation
        circle_x = int((self.screen_width // 2) - (horizontal_angle / self.HORIZONTAL_THRESHOLD) * (self.screen_width // 6))
        circle_y = int((self.screen_height // 2) - (vertical_angle / self.VERTICAL_THRESHOLD) * (self.screen_height // 6))
        circle_x = max(0, min(self.screen_width, circle_x))
        circle_y = max(0, min(self.screen_height, circle_y))

        # Determine which region the face is pointing towards
        region_x = 0 if circle_x < self.vertical_lines[0] else 1 if circle_x < self.vertical_lines[1] else 2
        region_y = 0 if circle_y < self.horizontal_lines[0] else 1 if circle_y < self.horizontal_lines[1] else 2

        # Highlight the region in the overlay
        cv2.rectangle(overlay, (region_x * self.screen_width // 3, region_y * self.screen_height // 3),
                      ((region_x + 1) * self.screen_width // 3, (region_y + 1) * self.screen_height // 3),
                      (0, 255, 0, 0.1), -1)

        # Determine the horizontal and vertical orientation of the face
        horizontal_orientation = "center"
        if horizontal_angle > self.HORIZONTAL_THRESHOLD:
            horizontal_orientation = "right"
        elif horizontal_angle < -self.HORIZONTAL_THRESHOLD:
            horizontal_orientation = "left"

        vertical_orientation = "center"
        if vertical_angle > self.VERTICAL_THRESHOLD:
            vertical_orientation = "down"
        elif vertical_angle < -self.VERTICAL_THRESHOLD:
            vertical_orientation = "up"

        # Combine the orientation information
        orientation = f"{horizontal_orientation}-{vertical_orientation}"

        # Display the orientation and ratio information on the frame
        cv2.putText(frame, f"H: {horizontal_angle:.2f}, V: {vertical_angle:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, orientation, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw a circle in the region the face is pointing towards
        cv2.circle(frame, (circle_x, circle_y), 10, (0, 0, 255), -1)

if __name__ == "__main__":
    tracker = FaceDirectionTracker()
    tracker.track()
