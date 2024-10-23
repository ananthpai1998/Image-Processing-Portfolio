import numpy as np
import cv2

def detect_motion(current_frame, previous_frame):
    try:
        # Convert frames to grayscale
        current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        # current_frame_gray = cv2.GaussianBlur(current_frame_gray, (5, 5), 0)
        # previous_frame_gray = cv2.GaussianBlur(previous_frame_gray, (5, 5), 0)

        # Calculate absolute difference
        abs_diff = cv2.absdiff(current_frame_gray, previous_frame_gray)

        # Apply threshold to the difference image
        _, thresholded_diff = cv2.threshold(abs_diff, 30, 255, cv2.THRESH_BINARY)

        # Use morphological operations to remove small noise (optional but improves results)
        kernel = np.ones((3, 3), np.uint8)
        cleaned_diff = cv2.morphologyEx(thresholded_diff, cv2.MORPH_OPEN, kernel)

        # Calculate the ratio of significant pixels
        significant_pixel_count = np.sum(cleaned_diff > 0)
        total_pixels = cleaned_diff.size
        motion_ratio = significant_pixel_count / total_pixels

        print(f"Motion Ratio: {motion_ratio}")

        # Return True if motion is significant (adjust threshold as needed)
        return motion_ratio > 0.0001

    except Exception as e:
        print(f"Error in motion detection: {str(e)}")
        return False

# Capture webcam video
cap = cv2.VideoCapture(0)

# Read the first frame
ret, previous_frame = cap.read()

try:
    while True:
        ret, current_frame = cap.read()
        if not ret:
            break

        # Detect motion
        motion_detected = detect_motion(current_frame, previous_frame)

        # Create a copy of the frame to avoid overwriting original with text
        frame_with_text = current_frame.copy()

        # Display the motion detection status on the frame
        text = f'Motion Detected: {motion_detected}'
        if motion_detected:
            cv2.putText(frame_with_text, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame_with_text, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # Display the current frame using OpenCV
        cv2.imshow('Webcam Feed', frame_with_text)

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Update previous frame
        previous_frame = current_frame

except KeyboardInterrupt:
    print("Exiting...")

finally:
    cap.release()
    cv2.destroyAllWindows()
