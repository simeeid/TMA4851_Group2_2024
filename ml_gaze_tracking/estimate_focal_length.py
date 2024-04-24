import cv2

def capture_photo(save_path='captured_photo.jpg'):
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Could not open webcam")
        return

    # Capture a single frame
    ret, frame = cap.read()

    # Check if the frame is captured correctly
    if ret:
        # Save the captured image to the specified path
        cv2.imwrite(save_path, frame)
        print(f"Photo captured and saved to {save_path}")
    else:
        print("Failed to capture photo")

    # Release the webcam
    cap.release()
    cv2.destroyAllWindows()

# Call the function to capture and save the photo
capture_photo('captured_photo.jpg')
