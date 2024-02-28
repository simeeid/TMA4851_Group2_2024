import cv2
import dlib
import numpy as np
from face import Face


class Tracker:
    def __init__(self, shared_data):
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")
        # Modelling tools
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('classes/shape_predictor_68_face_landmarks.dat')

        # Get screen dimensions
        self.screen_width = int(self.cap.get(3))
        self.screen_height = int(self.cap.get(4))
    
        self.vec_sacle = 5
        self.epsilon = 10   
        
        # self.vertical_angle = 0
        self.shared_data = shared_data                #tolerance for reaching target

    def get_vertical_angle(self):
        return self.vertical_angle

    def draw_vector(self, frame, start_point, direction_vector, scale=5, color=(0, 255, 0), thickness=2):
        # Draw a vector (line) on the frame to indicate direction
        end_point = (int(start_point[0] + direction_vector[0] * scale), int(start_point[1] + direction_vector[1] * scale))
        cv2.line(frame, start_point, end_point, color, thickness)

    def start_video(self):
        # Main tracking loop
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)          # grayscale image
            faces = self.detector(gray, 0)
            
            for face in faces:
                if face is not None:
                    face_obj = Face(frame,face, gray,self.predictor, self.screen_width, self.screen_height, method='improved')
                    forward_vector, nose_top = face_obj.vector, face_obj.nose_top
                    # self.vertical_angle = face_obj.vertical_angle
                    self.shared_data = face_obj.vertical_angle
                    # notice change
                    self.draw_vector(frame, np.array(nose_top), forward_vector, scale=self.vec_sacle, color=(255, 0, 0), thickness=2)
                    
                else :
                    print("No face detected")
                               

            match cv2.waitKey(1):   # Exit on ESC
                    case 27:
                        break


            cv2.imshow('Frame', frame)
        # Release and close
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    Tracker([]).start_video()
