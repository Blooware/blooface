import cv2
import time
import detector
from deepface import DeepFace


# Config deepface
model = 'Facenet'
detector_type = 'dlib'
distance_metric = 'euclidean'# 'euclidean_l2'

identity = ""
recognize = 0
cap = cv2.VideoCapture(0)

# font which we will be using to display FPS
font = cv2.FONT_HERSHEY_SIMPLEX

# used to record the time when we processed last frame
prev_frame_time = 0

# used to record the time at which we processed current frame
new_frame_time = 0

# Initialize the detector with the chosen model and resize value
det = detector.FrameDetector(detector_type, 1)

while True:
    # Read frame from cv2 source
    ret, img = cap.read()

    # Detect faces in frame
    det_img, detections = det.detect(img)



    # If faces are detected and recognize is enabled, recognize the face
    if len(detections) > 0 and recognize == 1 and ret:

        # For every detection, identify face
        for idx, detection in enumerate(detections):

            # # Get face from frame
            # crop_img = img[detection[1]:detection[1] +
            #                detection[3], detection[0]:detection[0]+detection[2]]

            # # Write face to file
            cv2.imwrite(str(idx) + '.jpg', img)

            # Recognize the face
            recognition = DeepFace.find(
                img_path=str(idx) + '.jpg',
                db_path="./dataset",
                model_name=model,
                enforce_detection=False,
                distance_metric=distance_metric,
                detector_backend=detector_type)

            if recognition.shape[0] > 0:
                identity = recognition.iloc[0].identity
                # identity = identity.split('/')[5]
                cv2.putText(img, identity, (detection[0], detection[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 2, cv2.LINE_AA)

            else:
                identity = "Unknown"

    elif len(detections) > 0:
        cv2.putText(img, 'FACE FOUND', (7, 70), font, .5, (100, 255, 0), 1, cv2.LINE_AA)
    else:
        identity = ""

    new_frame_time = time.time()
            
    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time

    # converting the fps into integer
    fps = int(fps)

    # converting the fps to string so that we can display it on frame
    # by using putText function
    fps = str(fps)

    # putting the FPS count on the frame
    cv2.putText(img, fps, (7, 70), font, .5, (100, 255, 0), 1, cv2.LINE_AA)

    # Add text to frame and show
    cv2.imshow('Input', img)

    # Wait for Esc press and close all windows
    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
