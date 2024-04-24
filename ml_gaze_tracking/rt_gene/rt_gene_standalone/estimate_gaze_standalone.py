#!/usr/bin/env python

# Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)

from __future__ import print_function, division, absolute_import

import argparse
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from rt_gene.extract_landmarks_method_base import LandmarkMethodBase
from rt_gene.gaze_tools import get_phi_theta_from_euler, limit_yaw
from rt_gene.gaze_tools_standalone import euler_from_matrix

script_path = os.path.dirname(os.path.realpath(__file__))


def load_camera_calibration(calibration_file):
    import yaml
    with open(calibration_file, 'r') as f:
        cal = yaml.safe_load(f)

    dist_coefficients = np.array(cal['distortion_coefficients']['data'], dtype='float32').reshape(1, 5)
    camera_matrix = np.array(cal['camera_matrix']['data'], dtype='float32').reshape(3, 3)

    return dist_coefficients, camera_matrix


def extract_eye_image_patches(subjects):
    for subject in subjects:
        le_c, re_c, _, _ = subject.get_eye_image_from_landmarks(subject, landmark_estimator.eye_image_size)
        subject.left_eye_color = le_c
        subject.right_eye_color = re_c


def estimate_gaze(base_name, color_img, dist_coefficients, camera_matrix):
    faceboxes = landmark_estimator.get_face_bb(color_img)
    if len(faceboxes) == 0:
        tqdm.write('Could not find faces in the image')
        return

    subjects = landmark_estimator.get_subjects_from_faceboxes(color_img, faceboxes)
    extract_eye_image_patches(subjects)

    input_r_list = []
    input_l_list = []
    input_head_list = []
    valid_subject_list = []

    for idx, subject in enumerate(subjects):
        if subject.left_eye_color is None or subject.right_eye_color is None:
            tqdm.write('Failed to extract eye image patches')
            continue

        success, rotation_vector, _ = cv2.solvePnP(landmark_estimator.model_points,
                                                   subject.landmarks.reshape(len(subject.landmarks), 1, 2),
                                                   cameraMatrix=camera_matrix,
                                                   distCoeffs=dist_coefficients, flags=cv2.SOLVEPNP_DLS)

        if not success:
            tqdm.write('Not able to extract head pose for subject {}'.format(idx))
            continue

        _rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        _rotation_matrix = np.matmul(_rotation_matrix, np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]))
        _m = np.zeros((4, 4))
        _m[:3, :3] = _rotation_matrix
        _m[3, 3] = 1
        # Go from camera space to ROS space
        _camera_to_ros = [[0.0, 0.0, 1.0, 0.0],
                          [-1.0, 0.0, 0.0, 0.0],
                          [0.0, -1.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 1.0]]
        roll_pitch_yaw = list(euler_from_matrix(np.dot(_camera_to_ros, _m)))
        roll_pitch_yaw = limit_yaw(roll_pitch_yaw)

        phi_head, theta_head = get_phi_theta_from_euler(roll_pitch_yaw)

        face_image_resized = cv2.resize(subject.face_color, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        head_pose_image = landmark_estimator.visualize_headpose_result(face_image_resized, (phi_head, theta_head))

        if args.vis_headpose:
            plt.axis("off")
            plt.imshow(cv2.cvtColor(head_pose_image, cv2.COLOR_BGR2RGB))
            plt.show()

        if args.save_headpose:
            # add idx to cope with multiple persons in one image
            cv2.imwrite(os.path.join(args.output_path, os.path.splitext(base_name)[0] + '_headpose_%s.jpg'%(idx)), head_pose_image)

        input_r_list.append(gaze_estimator.input_from_image(subject.right_eye_color))
        input_l_list.append(gaze_estimator.input_from_image(subject.left_eye_color))
        input_head_list.append([theta_head, phi_head])
        valid_subject_list.append(idx)

    if len(valid_subject_list) == 0:
        return

    gaze_est = gaze_estimator.estimate_gaze_twoeyes(inference_input_left_list=input_l_list,
                                                    inference_input_right_list=input_r_list,
                                                    inference_headpose_list=input_head_list)
    
    #print('GAZE EST: ', gaze_est)

    for subject_id, gaze, headpose in zip(valid_subject_list, gaze_est.tolist(), input_head_list):
        subject = subjects[subject_id]

        #print("headpose: ", headpose, "gaze: ", gaze)
        # Add head pose angles to gaze angles
        combined_gaze = [gaze[0] + headpose[0], gaze[1] + headpose[1]]

        # Build visualizations with the modified gaze angles
        r_gaze_img = gaze_estimator.visualize_eye_result(subject.right_eye_color, combined_gaze)
        l_gaze_img = gaze_estimator.visualize_eye_result(subject.left_eye_color, combined_gaze)
        s_gaze_img = np.concatenate((r_gaze_img, l_gaze_img), axis=1)

        if args.vis_gaze:
            #plt.axis("off")
            cv2.imshow('video', cv2.cvtColor(s_gaze_img, cv2.COLOR_BGR2RGB))
            #plt.show()

        if args.save_gaze:
            cv2.imwrite(os.path.join(args.output_path, os.path.splitext(base_name)[0] + '_gaze_%s.jpg'%(subject_id)), s_gaze_img)
            # cv2.imwrite(os.path.join(args.output_path, os.path.splitext(base_name)[0] + '_left.jpg'), subject.left_eye_color)
            # cv2.imwrite(os.path.join(args.output_path, os.path.splitext(base_name)[0] + '_right.jpg'), subject.right_eye_color)

        if args.save_estimate:
            # add subject_id to cope with multiple persons in one image
            with open(os.path.join(args.output_path, os.path.splitext(base_name)[0] + '_output_%s.txt'%(subject_id)), 'w+') as f:
                f.write(os.path.splitext(base_name)[0] + ', [' + str(headpose[1]) + ', ' + str(headpose[0]) + ']' +
                        ', [' + str(gaze[1]) + ', ' + str(gaze[0]) + ']' + ', [' + str(combined_gaze[1]) + ', ' + str(combined_gaze[0]) + ']' + '\n')

        return combined_gaze

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimate gaze from images')
    parser.add_argument('im_path', type=str, default=os.path.abspath(os.path.join(script_path, './samples_gaze/')),
                        nargs='?', help='Path to an image or a directory containing images')
    parser.add_argument('--calib-file', type=str, dest='calib_file', default=None, help='Camera calibration file')
    parser.add_argument('--vis-headpose', dest='vis_headpose', action='store_true', help='Display the head pose images')
    parser.add_argument('--no-vis-headpose', dest='vis_headpose', action='store_false', help='Do not display the head pose images')
    parser.add_argument('--save-headpose', dest='save_headpose', action='store_true', help='Save the head pose images')
    parser.add_argument('--no-save-headpose', dest='save_headpose', action='store_false', help='Do not save the head pose images')
    parser.add_argument('--vis-gaze', dest='vis_gaze', action='store_true', help='Display the gaze images')
    parser.add_argument('--no-vis-gaze', dest='vis_gaze', action='store_false', help='Do not display the gaze images')
    parser.add_argument('--save-gaze', dest='save_gaze', action='store_true', help='Save the gaze images')
    parser.add_argument('--save-estimate', dest='save_estimate', action='store_true', help='Save the predictions in a text file')
    parser.add_argument('--no-save-gaze', dest='save_gaze', action='store_false', help='Do not save the gaze images')
    parser.add_argument('--gaze_backend', choices=['tensorflow', 'pytorch'], default='tensorflow')
    parser.add_argument('--output_path', type=str, default=os.path.abspath(os.path.join(script_path, './samples_gaze/out')),
                        help='Output directory for head pose and gaze images')
    parser.add_argument('--models', nargs='+', type=str, default=[os.path.abspath(os.path.join(script_path, '../rt_gene/model_nets/Model_allsubjects1.h5'))],
                        help='List of gaze estimators')
    parser.add_argument('--device-id-facedetection', dest="device_id_facedetection", type=str, default='cuda:0', help='Pytorch device id. Set to "cpu:0" to disable cuda')

    parser.set_defaults(vis_gaze=True)
    parser.set_defaults(save_gaze=True)
    parser.set_defaults(vis_headpose=False)
    parser.set_defaults(save_headpose=True)
    parser.set_defaults(save_estimate=False)

    args = parser.parse_args()

    image_path_list = []
    if os.path.isfile(args.im_path):
        image_path_list.append(os.path.split(args.im_path)[1])
        args.im_path = os.path.split(args.im_path)[0]
    elif os.path.isdir(args.im_path):
        for image_file_name in sorted(os.listdir(args.im_path)):
            if image_file_name.lower().endswith('.jpg') or image_file_name.lower().endswith('.png') or image_file_name.lower().endswith('.jpeg'):
                if '_gaze' not in image_file_name and '_headpose' not in image_file_name:
                    image_path_list.append(image_file_name)
    else:
        tqdm.write('Provide either a path to an image or a path to a directory containing images')
        sys.exit(1)

    tqdm.write('Loading networks')
    landmark_estimator = LandmarkMethodBase(device_id_facedetection=args.device_id_facedetection,
                                            checkpoint_path_face=os.path.abspath(os.path.join(script_path, "../rt_gene/model_nets/SFD/s3fd_facedetector.pth")),
                                            checkpoint_path_landmark=os.path.abspath(
                                                os.path.join(script_path, "../rt_gene/model_nets/phase1_wpdc_vdc.pth.tar")),
                                            model_points_file=os.path.abspath(os.path.join(script_path, "../rt_gene/model_nets/face_model_68.txt")))

    if args.gaze_backend == "tensorflow":
        from rt_gene.estimate_gaze_tensorflow import GazeEstimator

        gaze_estimator = GazeEstimator("/gpu:0", args.models)
    elif args.gaze_backend == "pytorch":
        from rt_gene.estimate_gaze_pytorch import GazeEstimator

        gaze_estimator = GazeEstimator("cuda:0", args.models)
    else:
        raise ValueError("Incorrect gaze_base backend, choices are: tensorflow or pytorch")

    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)
    """
    for image_file_name in tqdm(image_path_list):
        tqdm.write('Estimate gaze on ' + image_file_name)
        image = cv2.imread(os.path.join(args.im_path, image_file_name))
        if image is None:
            tqdm.write('Could not load ' + image_file_name + ', skipping this image.')
            continue

        if args.calib_file is not None:
            _dist_coefficients, _camera_matrix = load_camera_calibration(args.calib_file)
        else:
            im_width, im_height = image.shape[1], image.shape[0]
            tqdm.write('WARNING!!! You should provide the camera calibration file, otherwise you might get bad results. Using a crude approximation!')
            _dist_coefficients, _camera_matrix = np.zeros((1, 5)), np.array(
                [[im_height, 0.0, im_width / 2.0], [0.0, im_height, im_height / 2.0], [0.0, 0.0, 1.0]])

        estimate_gaze(image_file_name, image, _dist_coefficients, _camera_matrix)
    """
    import dlib

    def find_face_detection_ROI_coords(image, roi_w_percent=100, roi_h_percent=100):
        roi_width = image.shape[1] * roi_w_percent // 100
        roi_height = image.shape[0] * roi_h_percent // 100
        roi_y_start = (image.shape[0] - roi_height) // 2
        roi_x_start = (image.shape[1] - roi_width) // 2
        return roi_x_start, roi_y_start, roi_width, roi_height
    
    def draw_ROI(image):
        roi_x_start, roi_y_start, roi_width, roi_height = find_face_detection_ROI_coords(image)
        roi_x_end = roi_x_start + roi_width
        roi_y_end = roi_y_start + roi_height
        cv2.rectangle(image, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), (255, 0, 0), 2)

    def find_eye_centers_image_frame(image):
        # Convert the whole image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Load the face detector and landmark predictor
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        # Detect faces in the whole image
        faces = detector(gray_image, 1)
        if len(faces) == 0:
            return None  # No faces detected in the image

        # Find the largest face in the image
        largest_face = max(faces, key=lambda rect: rect.width() * rect.height())
        landmarks = predictor(gray_image, largest_face)

        # Calculate the centers of the eyes
        left_eye_center = ((landmarks.part(36).x + landmarks.part(39).x) // 2, 
                           (landmarks.part(36).y + landmarks.part(39).y) // 2)
        right_eye_center = ((landmarks.part(42).x + landmarks.part(45).x) // 2, 
                            (landmarks.part(42).y + landmarks.part(45).y) // 2)

        return [left_eye_center, right_eye_center]

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

    def transform_centers_image_to_screen_frame(centers_image_frame, image_width, image_height, screen_pixel_w, screen_pixel_h):
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

    def transform_centers_image_to_camera_frame(centers_image_frame, focal_length, depth_estimate, im_w, im_h, screen_pixel_w, screen_pixel_h):
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
        
        return centers_camera_frame3

    def draw_eye_centers(image, eye_centers_screen_frame):
        for center in eye_centers_screen_frame:
            cv2.circle(image, center, 2, (0, 255, 0), -1)

    def draw_gaze_vector(image, eye_centers_image_frame, gaze_angles, length=100):
        # Assuming gaze_angles contains [theta, phi] in radians
        theta, phi = gaze_angles

        for center in eye_centers_image_frame:
            center_int = (int(center[0]), int(center[1]))
            # Calculate the end point of the gaze vector
            end_x = int(center_int[0] + length * np.cos(phi) * np.cos(theta))
            end_y = int(center_int[1] - length * np.sin(phi))

            cv2.line(image, center_int, (end_x, end_y), (255, 0, 0), 2)

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

    def find_dist_between_eyes(eye_centers):
        if eye_centers is None or len(eye_centers) < 2:
            return None
        dist_between_eyes = np.linalg.norm(np.array(eye_centers[0]) - np.array(eye_centers[1]))
        return dist_between_eyes

    #TODO fix this
    def estimate_distance_to_cam_camera_frame(dist_between_eyes_screen_frame, focal_length_image_pixels, avg_dist_mm):
        if dist_between_eyes_screen_frame is None:
            return None
        return focal_length_image_pixels * avg_dist_mm / dist_between_eyes_screen_frame

    def find_screen_gaze_point_camera_frame(eyes_middle_camera_frame, distance_to_camera_mm, gaze_angles):
        #angles are 0 when looking straight forward

        return distance_to_camera_mm*np.tan(gaze_angles) + eyes_middle_camera_frame

    def cam_to_screen_pixels(unit_ratio, offset, cam_gaze_point_2d): 
        x = unit_ratio*cam_gaze_point_2d[0] + offset[0]
        y = -unit_ratio*cam_gaze_point_2d[1] + offset[1]
        return (int(x),int(y))
    
    def resize_image(image, scale_percent):
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    def make_camera_matrix(img_w, img_h, focal_len_pixels):
        return np.array([[focal_len_pixels, 0, img_w/2], [0, focal_len_pixels, img_h/2], [0, 0, 1]])

    def display_screen_gaze_point(gaze_screen_pixel_coords, screen_pixel_w, screen_pixel_h):
        window_scale_percent = 20 
        # Create an image to represent the screen
        screen_rep = np.zeros((int(screen_pixel_h*window_scale_percent/100), int(screen_pixel_w*window_scale_percent/100), 3), dtype=np.uint8)

        scaled_down_coords = (int(gaze_screen_pixel_coords[0]*window_scale_percent/100), int(gaze_screen_pixel_coords[1]*window_scale_percent/100))
        # Mark the gaze point on the screen representation
        cv2.circle(screen_rep, scaled_down_coords, radius=10, color=(0, 255, 0), thickness=-1)

        # Display the screen representation in a window named 'Gaze Point on Screen'
        cv2.imshow('Gaze Point on Screen', screen_rep)

    def gaze_to_screen_pixel_coords(gaze_angles, eye_centers_image_frame, image, screen_pixel_w, screen_pixel_h):
        focal_len_image_pixels = 1317.74
        avg_between_eyes_dist_mm = 63
        screen_width_mm = 290.3
        screen_height_mm = 188.7
        screen_pixels_per_mm = 8.82
        cam_to_screen_pixels_unit_ratio = screen_pixels_per_mm
        #cam_to_screen_offset_mm = (screen_width_mm/2,0)
        cam_to_screen_offset_pixels = (screen_pixel_w/2,0)
        image_w = image.shape[1]
        image_h = image.shape[0]
        
        #print("eye_centers_image_frame: ", eye_centers_image_frame)
        #eye_centers_screen_frame = transform_centers_image_to_screen_frame(eye_centers_image_frame, 
        #                                                                   image_w, image_h, 
        #                                                                  screen_pixel_w, screen_pixel_h)
        #print(f"Eye centers in screen frame: {eye_centers_screen_frame}")
        
        dist_between_eyes_image_frame = find_dist_between_eyes(eye_centers_image_frame)
        #print(f"Distance between eyes in screen frame: {dist_between_eyes_screen_frame}")
        
        #TODO fix focal length
        distance_to_camera = estimate_distance_to_cam_camera_frame(dist_between_eyes_image_frame, focal_len_image_pixels, avg_between_eyes_dist_mm)
        #print(f"Estimated distance to camera: {distance_to_camera}")
        
        #TODO fix focal length
        eye_centers_camera_frame = transform_centers_image_to_camera_frame(eye_centers_image_frame, 
                                                                           focal_len_image_pixels, 
                                                                           distance_to_camera, 
                                                                           image_w, image_h, 
                                                                           screen_pixel_w, screen_pixel_h)
        #print(f"Eye centers in camera frame: {eye_centers_camera_frame}")

        eyes_middle_xy_camera_frame = np.mean(eye_centers_camera_frame, axis=0)
        #print(f"Middle point of eyes in camera frame: {eyes_middle_xy_camera_frame}")
        
        screen_gaze_point_camera_frame = find_screen_gaze_point_camera_frame(eyes_middle_xy_camera_frame, distance_to_camera, gaze_angles)
        #print(f"Screen gaze point in camera frame: {screen_gaze_point_camera_frame}")
        
        gaze_screen_pixel_coords = cam_to_screen_pixels(cam_to_screen_pixels_unit_ratio, cam_to_screen_offset_pixels, screen_gaze_point_camera_frame)
        #print(f"Pixel coordinates on screen: {gaze_screen_pixel_coords}")

        
        
        return gaze_screen_pixel_coords

     # Open a connection to the webcam
    
    cap = cv2.VideoCapture(0)
    while True:
        ret, image = cap.read()
        image = resize_image(image, 20)
        screen_pixel_w = 2560
        screen_pixel_h = 1664
        focal_len_image_pixels = 1317.74
        img_w = image.shape[1]
        img_h = image.shape[0]

        _dist_coefficients = np.zeros((1, 5))
        _camera_matrix = make_camera_matrix(img_w, img_h, focal_len_image_pixels)
        
        if ret:          
            print(f"Image shape: {image.shape}")
            gaze_angles = estimate_gaze('webcam_frame', image, _dist_coefficients, _camera_matrix)
            """
            gaze_angle[0] is horizontal, positive to the right
            gaze_angle[1] is vertical, positive upwards
            """

            eye_centers_image_frame = find_eye_centers_image_frame(image)

            if gaze_angles is None or len(gaze_angles) != 2 or eye_centers_image_frame is None:
                print("skipping frame, gaze_angles: ", gaze_angles, "eye_centers_image_frame: ", eye_centers_image_frame)
                continue

            draw_eye_centers(image, eye_centers_image_frame)
            #draw_gaze_vector(image, eye_centers_image_frame, gaze_angles)
            draw_ROI(image)
            gaze_screen_pixel_coords = gaze_to_screen_pixel_coords(gaze_angles, eye_centers_image_frame, image, screen_pixel_w, screen_pixel_h)
        

            #out of bounds warning
            if gaze_screen_pixel_coords[0] < 0 or gaze_screen_pixel_coords[0] > screen_pixel_w or gaze_screen_pixel_coords[1] < 0 or gaze_screen_pixel_coords[1] > screen_pixel_h:
                cv2.putText(image, "Gaze coordinates are out of screen bounds", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                cv2.putText(image, f"Pixel coords: x={gaze_screen_pixel_coords[0]}, y={gaze_screen_pixel_coords[1]}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
            
            #print info
            print(f"Gaze angles: {gaze_angles}")
            print(f"x={gaze_screen_pixel_coords[0]}, y={gaze_screen_pixel_coords[1]}")

            #visualize gaze point
            display_screen_gaze_point(gaze_screen_pixel_coords, screen_pixel_w, screen_pixel_h)

            #display video feed with eye positions
            cv2.imshow('image', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break                       
        else:
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

#TODO check if calibration is correct
#fix the focal length, pixels measured on screen is incorrect