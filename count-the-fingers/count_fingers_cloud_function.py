import numpy as np
import base64
import logging
from io import BytesIO
from PIL import Image
import cv2
import scipy.ndimage
from typing import Dict

# Set up logging for error tracking and debugging
logging.basicConfig(level=logging.INFO)

# Define a 3x3 structuring element for convolution used for detecting finger-like structures
finger_structuring_element = np.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
], dtype=np.uint8)

def identify_finger_like_objects(abs_frame: np.ndarray) -> np.ndarray:
    """
    Identify finger-like objects in the given frame based on convolution with a structuring element.
    Args:
        abs_frame (np.ndarray): The absolute difference frame where objects need to be identified.
    Returns:
        np.ndarray: The frame with bounding boxes drawn around detected objects.
    """
    # Apply convolution to the absolute frame with the finger structuring element
    convolution_result = scipy.ndimage.convolve(abs_frame, finger_structuring_element, mode='constant', cval=0)
    
    # Threshold for identifying regions of interest
    threshold = 50  
    finger_like_regions = (convolution_result >= threshold).astype(np.uint8) * 255
    
    # Label the connected components in the thresholded image
    labeled_objects, num_objects = scipy.ndimage.label(finger_like_regions)
    
    if num_objects > 0:
        # Find bounding boxes for each labeled object
        objects_boxes = scipy.ndimage.find_objects(labeled_objects)
        for obj_box in objects_boxes:
            if obj_box is not None:
                height = obj_box[0].stop - obj_box[0].start
                width = obj_box[1].stop - obj_box[1].start
                
                # Filter objects by size to match typical finger dimensions
                if (100 < height < 150) and (25 < width < 60):
                    # Draw bounding box around the detected object
                    cv2.rectangle(abs_frame, (obj_box[1].start, obj_box[0].start),
                                  (obj_box[1].stop, obj_box[0].stop), (255, 255, 255), 2)
                    # Annotate the bounding box with the height and width
                    cv2.putText(abs_frame, f'H: {height}, W: {width}', 
                                (obj_box[1].start, obj_box[0].start - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return abs_frame

def count(data: Dict[str, str]) -> str:
    """
    Compare two frames and detect finger-like objects. Returns a base64 encoded image showing the results.
    Args:
        data (Dict[str, str]): Dictionary containing base64-encoded 'current_frame' and 'previous_frame'.
    Returns:
        str: Base64-encoded image showing detected finger-like objects.
    """
    result = ''

    try:
        # Decode the base64-encoded images from the data and convert them to grayscale
        current_frame_img = Image.open(BytesIO(base64.b64decode(data['current_frame'].split(',')[1]))).convert('L')
        previous_frame_img = Image.open(BytesIO(base64.b64decode(data['previous_frame'].split(',')[1]))).convert('L')
        
        # Convert PIL images to numpy arrays
        current_frame = np.array(current_frame_img, dtype=np.float32)
        previous_frame = np.array(previous_frame_img, dtype=np.float32)
        
        # Create a buffer to store the output image
        buffered = BytesIO()
        
        # Ensure that both frames have the same dimensions before processing
        if current_frame.shape == previous_frame.shape:
            # Calculate the absolute difference between the current and previous frames
            abs_diff = np.abs(current_frame - previous_frame)
            
            # Identify finger-like objects in the difference image
            output_frame = identify_finger_like_objects(abs_diff)
            
            # Convert the output frame back to an image and save it as a JPEG
            img = Image.fromarray(np.uint8(output_frame))
            img.save(buffered, format="JPEG")
            
            # Encode the resulting image to base64
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            result = {'frame': 'data:image/jpeg;base64,' + img_base64,
                      'Description': 'Finger-like objects detected in the frame',
                      'utility1': False,
                      'utility2': 'Height'}
        else:
            logging.error("Frame dimensions do not match. Current frame: %s, Previous frame: %s",
                          current_frame.shape, previous_frame.shape)

    except Exception as e:
        # Log any errors that occur during the process
        logging.error("Error in count function: %s", str(e))
    
    return result