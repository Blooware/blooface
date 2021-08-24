import cv2
import detector
from deepface import DeepFace

identity = ""
recognize = 1
cap = cv2.VideoCapture(0)

# Initialize the detector with the chosen model and resize value
det = detector.FrameDetector('mtcnn', 0.5)

while True:
    # Read frame from cv2 source
    ret, img = cap.read()

    # Detect faces in frame
    img, detections = det.detect(img)

    # If faces are detected and recognize is enabled, recognize the face
    if len(detections) > 0 and recognize == 1 and ret:

        # For every detection, identify face
        for idx, detection in enumerate(detections):

            # Get face from frame
            crop_img = img[detection[1]:detection[1]+detection[3], detection[0]:detection[0]+detection[2]]


            # Write face to file
            cv2.imwrite(str(idx) + '.jpg', crop_img)

            # Recognize the face
            recognition = DeepFace.find(
                img_path = str(idx) + '.jpg', 
                db_path = "./dataset", 
                model_name = 'Facenet', 
                enforce_detection = False,
                distance_metric = 'euclidean',
                detector_backend = 'mtcnn')

            if recognition.shape[0] > 0:
                identity = recognition.iloc[0].identity
                # identity = identity.split('/')[5]
                cv2.putText(img, identity, (detection[0], detection[1]), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2, cv2.LINE_AA)
    
            else:
                identity = "Unknown"
        
    else:
        identity = ""

    # Add text to frame and show
    cv2.imshow('Input', img)

    # Wait for Esc press and close all windows
    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()