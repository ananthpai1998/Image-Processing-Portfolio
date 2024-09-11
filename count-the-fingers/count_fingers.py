import cv2
#I am using CV2 lib only for plotting boundary box around the fingers. All the finger detection logic is implemented using np or ndimageimport numpy as np.

import numpy as np
import scipy.ndimage

cap = cv2.VideoCapture(0)

if (cap.isOpened() is False):
    print("Unable to read camera feed")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter('output.avi',
                      cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                      10, (frame_width, frame_height))


# This function is used to convert color image into grayscale image
def get_grayscale(frame):     
    red = frame[:,:,0].astype('uint8')
    green = frame[:,:,1].astype('uint8')
    blue = frame[:,:,2].astype('uint8')
    grayscale = 0.21 * red + 0.72 * green + 0.07 * blue
    return grayscale.astype('uint8')

# This function will return the absolute difference between the frames
def get_absolute_diff(graysclae_frame_current, graysclae_frame_previous):
    graysclae_frame_current_normalized = (graysclae_frame_current - graysclae_frame_current.min()) / (graysclae_frame_current.max() - graysclae_frame_current.min()) * 255
    noisy_abs = np.abs(graysclae_frame_current_normalized - graysclae_frame_previous).astype('uint8')
    threshold = 30
    clean_abs = (noisy_abs > threshold).astype('uint8') * 255
    return clean_abs.astype('uint8')

finger_structuring_element = np.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
], dtype=np.uint8)

# This fuction with return the absolute difference frames, with boundary box around the fingers. Also it prints the finger count.
def identify_finger_like_objects(abs_frame):
    convolution_result = scipy.ndimage.convolve(abs_frame, finger_structuring_element, mode='constant', cval=0)
    threshold = 10  
    finger_like_regions = (convolution_result >= threshold).astype(np.uint8) * 255
    labeled_objects, num_objects = scipy.ndimage.label(finger_like_regions)
    if num_objects > 0:
        objects_boxes = scipy.ndimage.find_objects(labeled_objects)
        #initilizing counter to count the fingers
        count = 0
        for obj_box in objects_boxes:
            if obj_box is not None:
                height = obj_box[0].stop - obj_box[0].start
                width = obj_box[1].stop - obj_box[1].start
                if (height > 100 and height < 150) and (width > 30 and width < 60):
                    count += 1
                    cv2.rectangle(abs_frame, (obj_box[1].start, obj_box[0].start),
                                  (obj_box[1].stop, obj_box[0].stop), (255, 255, 255), 2)
                    cv2.putText(abs_frame, f'H: {height}, W: {width}', (obj_box[1].start, obj_box[0].start - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        print('Finger Count ', count)
    return abs_frame

graysclae_frame_previous = np.zeros((480, 640), dtype=np.uint8)

# Main loop
while(True):
    ret, frame = cap.read()
    if ret:
        graysclae_frame_current = get_grayscale(frame)
        abs_diff_frame = get_absolute_diff(graysclae_frame_current, graysclae_frame_previous)
        abs_frame_with_hand_detection = identify_finger_like_objects(abs_diff_frame)
        out.write(abs_frame_with_hand_detection)
        cv2.imshow('Frame', abs_frame_with_hand_detection)
        graysclae_frame_previous = graysclae_frame_current
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()