import cv2
import dlib
import numpy as np

# Initialize Dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def pupil_detection_from_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    for face in faces:
        landmarks = predictor(gray, face)

        # Assuming landmarks 36-41 correspond to the left eye and 42-47 to the right eye
        left_eye_pts = np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in range(36, 42)])
        right_eye_pts = np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in range(42, 48)])

        # Compute the bounding box for each eye
        le_x, le_y, le_w, le_h = cv2.boundingRect(left_eye_pts)
        re_x, re_y, re_w, re_h = cv2.boundingRect(right_eye_pts)

        # Crop the eye regions from the grayscale image
        left_eye_img = gray[le_y:le_y+le_h, le_x:le_x+le_w]
        right_eye_img = gray[re_y:re_y+re_h, re_x:re_x+re_w]

        # Detect the darkest point in the cropped eye images as the pupil location
        minVal, _, minLoc_left, _ = cv2.minMaxLoc(left_eye_img)
        minVal, _, minLoc_right, _ = cv2.minMaxLoc(right_eye_img)

        # Convert minLoc to the original image coordinate system
        left_pupil_global = (minLoc_left[0] + le_x, minLoc_left[1] + le_y)
        right_pupil_global = (minLoc_right[0] + re_x, minLoc_right[1] + re_y)

        # Visualization
        cv2.circle(image, left_pupil_global, 3, (255, 0, 0), -1)
        cv2.circle(image, right_pupil_global, 3, (255, 0, 0), -1)
    
    return image

# Load an image
image = cv2.imread("path_to_your_image.jpg")


cap = cv2.VideoCapture(0)
   
while True:
    # Capture frame-by-frame
    ret, image = cap.read()

    # Detect and visualize pupils
    result_image = pupil_detection_from_landmarks(image)

    # Display the result
    cv2.imshow("Pupil Detection", result_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.waitKey(0)
cv2.destroyAllWindows()
