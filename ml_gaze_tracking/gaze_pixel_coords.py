import numpy as np
import cv2
import os

est_focal_len_pixels = 1317.74

avg_between_eyes_dist_mm = 63

screen_width_mm = 290.3
screen_height_mm = 188.7

screen_pixel_w = 2560
screen_pixel_h = 1664

pixels_per_mm = 8.82
cam_to_screenpix_unit_ratio = pixels_per_mm

cam_to_screen_offset_mm = (screen_width_mm/2,0)
cam_to_screen_offset_pixels = (screen_pixel_w/2,0)

gaze_angles = (-0.1,-0.1)

import dlib

#three different frames of reference
#1. camera frame (x,y,z) mm
#2. screen frame (x,y) pixels
#3. image frame (x,y) pixels

def find_eye_centers_image_frame(image):
    height, width = image.shape[:2]

    # Load the face detector
    detector = dlib.get_frontal_face_detector()
    # Load the facial landmark predictor
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray_frame, 1)
    if len(faces) == 0:
        return None  # No faces detected

    # Find the largest face in the image
    largest_face = max(faces, key=lambda rect: rect.width() * rect.height())
    landmarks = predictor(gray_frame, largest_face)
    left_eye_center = ((landmarks.part(36).x + landmarks.part(39).x) // 2, (landmarks.part(36).y + landmarks.part(39).y) // 2)
    right_eye_center = ((landmarks.part(42).x + landmarks.part(45).x) // 2, (landmarks.part(42).y + landmarks.part(45).y) // 2)
    centers_image_frame = [left_eye_center, right_eye_center]

    return centers_image_frame

def find_eye_centers_image_frame_haarcascade(image):
    # Load the Haar Cascade for eye detection
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray_frame, 1.3, 5)
    
    centers = []
    for (ex, ey, ew, eh) in eyes:
        center_x = ex + ew // 2
        center_y = ey + eh // 2
        centers.append((center_x, center_y))
    
    # If no eyes or only one eye is detected, return None
    if len(centers) < 2:
        return None
    
    # Assuming the first two detections are the eyes
    return centers[:2]


def transform_centers_image_to_screen_frame(centers_image_frame, image_width, image_height ):
    #TODO fix this, add y offset
    #transform from image to screen frame
    scale_x = screen_pixel_w / image_width
    scale_y = screen_pixel_h / image_height

    y_offset = - screen_pixel_h/2
        
    eye_centers_screen_frame = [(screen_pixel_w - (center[0] * scale_x), (screen_pixel_h - center[1] * scale_y + y_offset)) for center in centers_image_frame]

    return eye_centers_screen_frame

def transform_centers_screen_to_camera_frame(centers_screen_frame, cam_to_screenpix_unit_ratio, cam_to_screen_offset_pixels,):
    eye_centers_camera_frame = []
    
    # Loop through each center in the screen frame
    for center in centers_screen_frame:
        # Calculate the x coordinate in the camera frame
        x_camera_frame = (center[0] - cam_to_screen_offset_pixels[0]) / cam_to_screenpix_unit_ratio
        # Calculate the y coordinate in the camera frame
        y_camera_frame = (center[1] - cam_to_screen_offset_pixels[1]) / cam_to_screenpix_unit_ratio
        # Append the transformed center to the list
        eye_centers_camera_frame.append((x_camera_frame, y_camera_frame))
    
    # Return the list of transformed centers
    return eye_centers_camera_frame

def transform_centers_image_to_camera_frame(centers_image_frame, focal_length, depth_estimate, im_w, im_h):

    centers_camera_frame = []
    for center in centers_image_frame:
        # Convert image frame coordinates to camera frame coordinates
        # Taking into account that the image x values are flipped

        x = center[0] * (screen_pixel_w / im_w)
        x = screen_pixel_w - x
        x -= screen_pixel_w / 2

        y = center[1] * (screen_pixel_h / im_h) - screen_pixel_h / 2

        y = -y #because iamge and screen coords have y in opposite dir

        x_camera_frame = x * (depth_estimate / focal_length)
        y_camera_frame = y * (depth_estimate / focal_length)
        centers_camera_frame.append((x_camera_frame, y_camera_frame))
    
    return centers_camera_frame


def display_eye_centers(image, eye_centers_screen_frame):
    if eye_centers_screen_frame is None or len(eye_centers_screen_frame) < 2:
        return None
    for center in eye_centers_screen_frame:
        cv2.circle(image, center, 2, (0, 255, 0), -1)
    cv2.imshow('Eyes Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def find_eyes_middle_xy_camera_frame(eye_centers_camera_frame):
    if eye_centers_camera_frame is None or len(eye_centers_camera_frame) < 2:
        return None
    eyes_middle_xy_camera_frame = np.mean(eye_centers_camera_frame, axis=0)
    return eyes_middle_xy_camera_frame

"""
def find_eye_centers_xy_camera_frame(image):
    eye_centers_image_frame = find_eye_centers_screen_frame(image)
    eye_centers_screen_frame = transform_centers_screen_frame(eye_centers_image_frame, image.shape[1], image.shape[0])
    eye_centers_camera_frame = transform_centers_camera_frame(eye_centers_screen_frame, gaze_angles)
    return eye_centers_camera_frame
"""

def find_dist_between_eyes_screen_frame(eye_centers_screen_frame):
    if eye_centers_screen_frame is None or len(eye_centers_screen_frame) < 2:
        return None
    dist_between_eyes_screen_frame = np.linalg.norm(np.array(eye_centers_screen_frame[0]) - np.array(eye_centers_screen_frame[1]))
    return dist_between_eyes_screen_frame

def estimate_distance_to_cam_camera_frame(dist_between_eyes_screen_frame, focal_len, avg_dist_mm):
    if dist_between_eyes_screen_frame is None:
        return None
    return focal_len * avg_dist_mm / dist_between_eyes_screen_frame

def find_screen_gaze_point_camera_frame(eyes_middle_camera_frame, distance_to_camera_mm, gaze_angles):
    return distance_to_camera_mm*np.tan(gaze_angles) + eyes_middle_camera_frame

def cam_to_screen_pixels(unit_ratio, offset, cam_gaze_point_2d): 
    x = unit_ratio*cam_gaze_point_2d[0] + offset[0]
    y = -unit_ratio*cam_gaze_point_2d[1] + offset[1]
    return (int(x),int(y))



def capture_photo():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam")
        return

    ret, image = cap.read()

    # Check if the image is captured correctly
    #if ret:
        # Save the captured image to the specified path
        #cv2.imshow("eyes", image)
   
    cap.release()
    return image

image = capture_photo()

print("Image height: ", image.shape[0], "Image width: ", image.shape[1])
print("Camera to screen offset pixels: ", cam_to_screen_offset_pixels)

if find_eye_centers_image_frame(image) is None:
    print("No eyes detected")
    exit()

eye_centers_image_frame = find_eye_centers_image_frame(image)
print("Eye centers in image frame: ", eye_centers_image_frame)

eye_centers_screen_frame = transform_centers_image_to_screen_frame(eye_centers_image_frame, image.shape[1], image.shape[0])
print("Eye centers in screen frame: ", eye_centers_screen_frame)

dist_between_eyes_screen_frame = find_dist_between_eyes_screen_frame(eye_centers_screen_frame)
print(f"Distance between eyes in screen frame pixels: {dist_between_eyes_screen_frame}")

distance_to_camera = estimate_distance_to_cam_camera_frame(dist_between_eyes_screen_frame, est_focal_len_pixels, avg_between_eyes_dist_mm)
print(f"Estimated distance to camera mm: {distance_to_camera}")

eye_centers_camera_frame = transform_centers_image_to_camera_frame(eye_centers_image_frame, est_focal_len_pixels, distance_to_camera, image.shape[1], image.shape[0])
print("Eye centers in camera frame: ", eye_centers_camera_frame)

eyes_middle_xy_camera_frame = np.mean(eye_centers_camera_frame, axis=0)
print("Middle point between eyes in camera frame mm: ", eyes_middle_xy_camera_frame)

screen_gaze_point_camera_frame = find_screen_gaze_point_camera_frame(eyes_middle_xy_camera_frame, distance_to_camera, gaze_angles)
print(f"Screen gaze point in camera frame mm: {screen_gaze_point_camera_frame}")

pixel_coords = cam_to_screen_pixels(cam_to_screenpix_unit_ratio, cam_to_screen_offset_pixels, screen_gaze_point_camera_frame)
print(f"Screen pixel coordinates: {pixel_coords}")
"""
screen_pixel_w = 2560
screen_pixel_h = 1664
"""

import cv2
import numpy as np

def display_gaze_point(pixel_coords):
    """
    Display the gaze point as a point on a scaled down display window.
    
    Args:
    - pixel_coords: The pixel coordinates where the gaze point should be displayed.
    """
    # Scale down dimensions to half the size
    scaled_screen_pixel_h = screen_pixel_h // 2
    scaled_screen_pixel_w = screen_pixel_w // 2
    
    # Create a white image of scaled down size
    display_img = np.ones((scaled_screen_pixel_h, scaled_screen_pixel_w, 3), dtype=np.uint8) * 255
    
    # Adjust pixel_coords to scaled down size
    scaled_pixel_coords = (int(pixel_coords[0] // 2), int(pixel_coords[1] // 2))
    
    # Draw a circle at the scaled gaze point
    cv2.circle(display_img, scaled_pixel_coords, 5, (0, 255, 0), -1)
    
    # Display the scaled image
    cv2.imshow("Gaze Point", display_img)
    cv2.waitKey(1)

def display_image_with_eye_centers(image, eye_centers):
    """
    Display the image with eye centers marked.
    
    Args:
    - image: The image where the eyes were detected.
    - eye_centers: The coordinates of the eye centers in the image frame.
    """
    for center in eye_centers:
        # Draw a circle at each eye center
        cv2.circle(image, tuple(int(x) for x in center), 2, (0, 0, 255), -1)
    
    # Display the image with eye centers marked
    cv2.imshow("Image with Eye Centers", image)
    cv2.waitKey(1)


def main_loop():
    """
    Main loop to continuously capture images and display the gaze point.
    """
    cap = cv2.VideoCapture(0)  # Open the default camera


    while True:
        ret, image = cap.read()

        image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))

        if find_eye_centers_image_frame(image) is not None:
            use_dlib = True  # Toggle this to False to use Haarcascades instead of dlib
            
            if use_dlib:
                eye_centers_image_frame = find_eye_centers_image_frame(image)
            else:
                eye_centers_image_frame = find_eye_centers_image_frame_haarcascade(image)
                display_image_with_eye_centers(image, eye_centers_image_frame)

            eye_centers_screen_frame = transform_centers_image_to_screen_frame(eye_centers_image_frame, image.shape[1], image.shape[0])
            
            dist_between_eyes_screen_frame = find_dist_between_eyes_screen_frame(eye_centers_screen_frame)
            
            distance_to_camera = estimate_distance_to_cam_camera_frame(dist_between_eyes_screen_frame, est_focal_len_pixels, avg_between_eyes_dist_mm)
            
            eye_centers_camera_frame = transform_centers_image_to_camera_frame(eye_centers_image_frame, est_focal_len_pixels, distance_to_camera, image.shape[1], image.shape[0])

            eyes_middle_xy_camera_frame = np.mean(eye_centers_camera_frame, axis=0)
            
            screen_gaze_point_camera_frame = find_screen_gaze_point_camera_frame(eyes_middle_xy_camera_frame, distance_to_camera, gaze_angles)
            pixel_coords = cam_to_screen_pixels(cam_to_screenpix_unit_ratio, cam_to_screen_offset_pixels, screen_gaze_point_camera_frame)
            
            display_gaze_point(pixel_coords)
        else:
            print("No eyes detected")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main_loop()
