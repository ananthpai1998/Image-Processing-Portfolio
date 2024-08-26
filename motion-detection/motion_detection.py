import numpy as np
import base64
import logging
from io import BytesIO
from PIL import Image
from typing import Dict




def detect_motion(data: Dict[str, str]) -> str:
    result = ''
    isMotionDetected = False
    try:
        current_frame_img = Image.open(BytesIO(base64.b64decode(data['current_frame'].split(',')[1]))).convert('L')
        previous_frame_img = Image.open(BytesIO(base64.b64decode(data['previous_frame'].split(',')[1]))).convert('L')
        
        current_frame = np.array(current_frame_img, dtype=np.float32)
        previous_frame = np.array(previous_frame_img, dtype=np.float32)
        
        buffered = BytesIO()
        
        if current_frame.shape == previous_frame.shape:
            abs_diff = np.abs(current_frame - previous_frame)
            
            threshold = 30  
            significant_pixel_count = np.sum(abs_diff > threshold)
            total_pixels = abs_diff.size
            print(significant_pixel_count / total_pixels)
            if significant_pixel_count / total_pixels > 0.005:
                isMotionDetected = True
            
            img = Image.fromarray(np.uint8(abs_diff))
            img.save(buffered, format="JPEG")
            
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            result = {'frame': data['current_frame'],
                      'Description': '',
                      'utility1': isMotionDetected,
                      'utility2': 'Height'}
        else:
            logging.error("Frame dimensions do not match. Current frame: %s, Previous frame: %s",
                          current_frame.shape, previous_frame.shape)

    except Exception as e:
        logging.error("Error in count function: %s", str(e))
    
    return result