import cv2
import dlib
import numpy as np


class Face:
    def __init__(self,frame,face,gray,predictor):
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        self.center = (int(x), int(y))
        self.landmarks = predictor(gray, face)
        self.draw_landmarks(frame)
        self.vector, self.nose_top = self.get_face_data()

        cv2.circle(frame, (self.center), 10, (0, 0, 255), -1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)            


    def draw_landmarks(self, frame):
        for n in range(0, 68):
            x = self.landmarks.part(n).x
            y = self.landmarks.part(n).y
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
    
    def get_face_data(self):
        # Calculate forward vector from ear to nose top
        left_ear = np.array([self.landmarks.part(0).x, self.landmarks.part(0).y])
        right_ear = np.array([self.landmarks.part(16).x, self.landmarks.part(16).y])
        ear_midpoint = (left_ear + right_ear) / 2
        nose_top = np.array([self.landmarks.part(27).x, self.landmarks.part(27).y])
        forward_vector = nose_top - ear_midpoint
        return forward_vector,nose_top