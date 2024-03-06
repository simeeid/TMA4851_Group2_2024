import cv2
import dlib
import numpy as np
from filter import LowPassFilter
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
    
        self.vec_scale = 1
        self.epsilon = 10   
        
        self.shared_data = shared_data
        self.running = True
        self.filter = LowPassFilter(0.1)

    def get_vertical_angle(self):
        return self.vertical_angle

    def draw_vector(self, frame, start_point, direction_vector, scale=5, color=(0, 255, 0), thickness=2):
        # Draw a vector (line) on the frame to indicate direction
        end_point = (int(start_point[0] + direction_vector[0] * scale), int(start_point[1] + direction_vector[1] * scale))
        cv2.line(frame, start_point, end_point, color, thickness)

    def start_video(self):
        # Main tracking loop
        translation_filter = LowPassFilter(0.4)
        rotation_filter = LowPassFilter(0.4)

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)          # grayscale image
            faces = self.detector(gray, 0)
            
            for face in faces:
                if face is not None:
                    face_obj = Face(frame,face, gray,self.predictor, self.screen_width, self.screen_height, method='improved', translation_filter=translation_filter, rotation_filter=rotation_filter)
                    forward_vector, nose_top = face_obj.vector, face_obj.nose_top
                    # self.vertical_angle = face_obj.vertical_angle
                    # smoothed_forward_vector = self.filter.apply_filter(forward_vector) # applying the filter
                    self.shared_data = face_obj.vertical_angle

                    # self.shared_data = -np.arcsin(smoothed_forward_vector[1] / 1000) * 180 / np.pi
                    # print(self.shared_data)
                    # notice change
                    self.draw_vector(frame, np.array(nose_top), forward_vector, scale=self.vec_scale, color=(255, 0, 0), thickness=2)
                    
                else :
                    print("No face detected")
                               

            match cv2.waitKey(1):   # Exit on ESC
                    case 27:
                        self.running = False
                        


            cv2.imshow('Frame', frame)
            
        # Release and close
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    Tracker(0).start_video()
