from io import BytesIO  # Used to handle binary data in memory as a stream
from PIL import Image  # Python Imaging Library, used to handle image manipulations
import base64  # Provides functions for encoding and decoding binary data into base64 format
import numpy as np  # Used for handling arrays and numerical operations
from typing import Dict  # Type hinting to specify a dictionary input type
import mediapipe as mp  # MediaPipe, a framework for machine learning pipelines like hand tracking

def track(data: Dict[str, str]) -> str:
    """
    Processes the input image from base64 string to detect hands using MediaPipe, 
    and returns the processed image with hand landmarks drawn on it in base64 format.

    Args:
    data (Dict[str, str]): A dictionary containing a base64-encoded image string under the key 'current_frame'.

    Returns:
    str: A dictionary containing the modified frame (with hand landmarks) as a base64-encoded JPEG image,
         along with some additional information in a dictionary format.
    """
    result = ''  # Initialize an empty result to hold the final processed frame
    try:
        # Decode the base64 image, remove the header ('data:image/jpeg;base64,'), and convert it to a PIL image
        current_frame_img = Image.open(BytesIO(base64.b64decode(data['current_frame'].split(',')[1])))
        
        # Convert the image to an RGB NumPy array for further processing
        current_frame = np.array(current_frame_img.convert('RGB')).astype(np.uint8)
        
        # Initialize MediaPipe's hand detection and drawing utilities
        mp_hands = mp.solutions.hands  # Hand detection model
        mp_drawing = mp.solutions.drawing_utils  # Drawing utilities for hand landmarks
        
        # Use the Hands module to process the frame and detect hands
        with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands:
            result = hands.process(current_frame)  # Process the frame for hand landmarks

            # If hand landmarks are detected, draw them on the current frame
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(current_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Convert the modified frame back to a PIL image
        img = Image.fromarray(current_frame)
        
        # Save the image in memory as a JPEG file
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        
        # Encode the modified image as base64
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Prepare the final result containing the image and additional info
        result = {'frame': 'data:image/jpeg;base64,' + img_base64,
                  'Description': '',  # Empty description placeholder
                  'utility1': False,  # Placeholder for utility1 flag
                  'utility2': False}  # Placeholder for utility2 flag
        
        # Close the resources
        current_frame_img.close()  # Close the image file
        buffered.close()  # Close the buffered stream
        
    except Exception as e:
        # Handle any errors that occur during the process
        print("An Error Occurred: ", str(e))

    return result  # Return the final processed result
