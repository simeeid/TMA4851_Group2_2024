import cv2
import dlib
import numpy as np

class Face:
    def __init__(self, frame, face, gray, predictor, screen_width, screen_height, method='improved'):
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        self.center = (int(x), int(y))
        self.landmarks = predictor(gray, face)
        self.draw_landmarks(frame)

        # Camera internals
        self.focal_length = screen_width
        self.center = (screen_width / 2, screen_height / 2)
        self.camera_matrix = np.array(
            [[self.focal_length, 0, self.center[0]],
             [0, self.focal_length, self.center[1]],
             [0, 0, 1]], dtype="double"
        )
        self.dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        
        # Choose the calculation method
        if method == 'simple':
            self.vector, self.nose_top, self.horizontal_angle, self.vertical_angle = self.get_face_data_simple()
        elif method == 'improved':
            self.vector, self.nose_top, self.horizontal_angle, self.vertical_angle = self.get_face_data_improved()

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)            

    def draw_landmarks(self, frame):
        for n in range(0, 68):
            x = self.landmarks.part(n).x
            y = self.landmarks.part(n).y
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
    
    def get_face_data_simple(self, scale=100):
        # Calculate forward vector from ear to nose top
        left_ear = np.array([self.landmarks.part(0).x, self.landmarks.part(0).y])
        right_ear = np.array([self.landmarks.part(16).x, self.landmarks.part(16).y])
        ear_distance = np.linalg.norm(right_ear - left_ear)
        ear_midpoint = (left_ear + right_ear) / 2
        nose_top = np.array([self.landmarks.part(27).x, self.landmarks.part(27).y])
        forward_vector = nose_top - ear_midpoint

        # Calculate orientation ratios
        horizontal_angle = - scale * forward_vector[0] / ear_distance
        vertical_angle = - scale * forward_vector[1] / ear_distance

        return forward_vector, nose_top, horizontal_angle, vertical_angle
    
    def get_face_data_improved(self):
        # 3D model points.
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])

        # 2D image points. If you change the image, you need to change vector
        image_points = np.array([
            (self.landmarks.part(30).x, self.landmarks.part(30).y), # Nose tip
            (self.landmarks.part(8).x, self.landmarks.part(8).y),   # Chin
            (self.landmarks.part(36).x, self.landmarks.part(36).y), # Left eye left corner
            (self.landmarks.part(45).x, self.landmarks.part(45).y), # Right eye right corner
            (self.landmarks.part(48).x, self.landmarks.part(48).y), # Left Mouth corner
            (self.landmarks.part(54).x, self.landmarks.part(54).y)  # Right Mouth corner
        ], dtype="double")

        # Solve for pose
        _, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, self.camera_matrix, self.dist_coeffs)

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

        return direction_vector, p1, horizontal_angle, vertical_angle