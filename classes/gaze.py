import cv2
import dlib
import numpy as np
from math import hypot
import time


class Gaze:
    def __init__(self, detection):
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")
        # Modelling tools
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('./classes/shape_predictor_68_face_landmarks.dat')

        # Get screen dimensions
        self.screen_width = int(self.cap.get(3))
        self.screen_height = int(self.cap.get(4))

        self.ear_values = {'Left': {'Up': 3.3, 'Center': 3.5, 'Down': 3.9}, 'Right': {'Up': 3.3, 'Center': 3.5, 'Down': 3.9}}
        
        self.ear_history = [0] * 10
        self.look_down_start_time, self.look_up_start_time = None, None

        # detection 0 if looking center, 1 if looking down and 2 for looking up 
        self.detection = detection         

        self.running = True

    def get_vertical_angle(self):
        return self.vertical_angle

    def eye_tracking(self, frame, eye_region, gray):
        height, width, _ = frame.shape
        mask = np.zeros((height, width), np.uint8)
        cv2.polylines(mask, [eye_region], True, 0, 2) 
        cv2.fillPoly(mask, [eye_region], 255)
        eye = cv2.bitwise_and(gray, gray, mask=mask)
        eye = cv2.GaussianBlur(eye, (7,7), 0)

        _, threshold = cv2.threshold(eye, 70, 255, cv2.THRESH_BINARY_INV) 
        threshold = cv2.bitwise_and(threshold, threshold, mask=mask)
        cv2.polylines(threshold, [eye_region], True, 0, 2) 
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True) 
        
        return eye, contours

    def calculate_ear(self, eye_region):
        left_point, right_point = eye_region[0], eye_region[3]
        center_top = np.mean(eye_region[1:3], axis=0).astype(int)
        center_bottom = np.mean(eye_region[4:6], axis=0).astype(int)

        horiz_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
        vert_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

        ear = horiz_line_length / vert_line_length
        return ear
    
    def calculate_thresholds(self, values, direction, lim):
        return values['Center'] + (values[direction] - values['Center']) / lim
    
    def display_calibration_message(self, frame, target, message, duration):
        circle_radius = 10
        circle_color = (255, 255, 0)
        text_color = (0, 0, 0)
        font_scale = 0.5
        font_thickness = 1

        cv2.circle(frame, target, circle_radius, circle_color, -1)
        cv2.putText(frame, message, (target[0] + circle_radius + 10, target[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)
        cv2.imshow("Calibration", frame)
        cv2.waitKey(duration)

    def calibration(self, camera, height, width):
        calibration_points = { 'Up': (int(width/2), 10), 'Center': (int(width/2), int(height/2)), 'Down': (int(width/2), int(height)-10)}
        camera = cv2.VideoCapture(0)

        for direction, target in calibration_points.items():
            left_ear, right_ear = [], []

            _, frame = camera.read()
            cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            self.display_calibration_message(frame, target, "LOOK HERE", duration=1000)
            cv2.imshow("Calibration", frame)
            cv2.waitKey(500)

            _, frame = camera.read()
            cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            self.display_calibration_message(frame, target, "Calibrating...", duration=2000)
            cv2.imshow("Calibration", frame)
            cv2.waitKey(1000)

            for _ in range(30):
                _, frame = camera.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.detector(gray)

                for face in faces:
                    shape = self.predictor(gray, face)
                    landmarks = np.array([[p.x, p.y] for p in shape.parts()])

                    left_eye_region, right_eye_region = landmarks[36:42], landmarks[42:48]
                    left_ear.append(self.calculate_ear(left_eye_region))
                    right_ear.append(self.calculate_ear(right_eye_region))



            average_left_ear, average_right_ear = np.mean(left_ear), np.mean(right_ear)
            self.ear_values['Left'][direction], self.ear_values['Right'][direction] = average_left_ear, average_right_ear

        cv2.destroyWindow("Calibration")

        return self.ear_values

    def start_video(self):
        # Main tracking loop
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break


            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)
            
            for face in faces:
                if face is not None:
                    shape = self.predictor(gray, face)
                    landmarks = np.array([[p.x, p.y] for p in shape.parts()])

                    left_eye_region = landmarks[36:42]
                    right_eye_region = landmarks[42:48]

                    self.eye_left, self.contours_left = self.eye_tracking(frame, left_eye_region, gray)
                    self.eye_right, self.contours_right = self.eye_tracking(frame, right_eye_region, gray)

                    ear_left, ear_right = self.calculate_ear(left_eye_region), self.calculate_ear(right_eye_region)

                    ear_value = (ear_left+ear_right)/2

                    self.ear_history.pop(0), self.ear_history.append(ear_value)
                    
                    threshold_left_up, threshold_right_up = self.calculate_thresholds(self.ear_values['Left'], 'Up', 2), self.calculate_thresholds(self.ear_values['Right'], 'Up', 2)
                    threshold_left_down, threshold_right_down = self.calculate_thresholds(self.ear_values['Left'], 'Down', 2), self.calculate_thresholds(self.ear_values['Right'], 'Down', 2)

                    threshold_up = (threshold_left_up+threshold_right_up)/2
                    threshold_down = (threshold_left_down+threshold_right_down)/2

                    if sum(ear > threshold_down for ear in self.ear_history) >= 5:
                        if self.look_down_start_time is None:
                            self.look_down_start_time = time.time()
                            self.look_up_start_time = None
                        else:
                            if time.time() - self.look_down_start_time >= 1:
                                self.detection = 1
                    elif sum(ear < threshold_up for ear in self.ear_history) >= 5:
                        if self.look_up_start_time is None:
                            self.look_up_start_time = time.time()
                            self.look_down_start_time = None
                        else:
                            if time.time() - self.look_up_start_time >= 1:
                                self.detection = 2
                    else:
                        self.look_down_start_time = None
                        self.detection = 0    


                else :
                    print("No face detected")

            cv2.putText(frame, "Press Enter to start calibration", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
            # cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            if self.detection == 1:
                cv2.putText(frame, "DOWN", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            elif self.detection == 2:
                cv2.putText(frame, "UP", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


            match cv2.waitKey(1):   # Exit on ESC
                case 13:  
                    self.ear_values = self.calibration(self.cap, self.screen_height, self.screen_width)
                    self.cap.release()
                    self.cap = cv2.VideoCapture(0)
                case 27:
                    self.running = False
                    

            cv2.imshow('Frame', frame)
            
        # Release and close
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    Gaze(0).start_video()
    