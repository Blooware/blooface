import cv2
import dlib 
from mtcnn import MTCNN

class FrameDetector:
    def __init__(self, mode, resize):
        self.mode = mode
        self.resize = resize
        self.dlib_detector = dlib.get_frontal_face_detector()
        self.mtcnn_detector = MTCNN()

    def detect(self, frame):
        frame = cv2.resize(frame, None, fx=self.resize, fy=self.resize, interpolation=cv2.INTER_AREA)
        all_detections = []
        if self.mode == 'dlib':
            detections = self.dlib_detector(frame, 1)
            
            for idx, detection in enumerate(detections):
                x = detection.left(); right = detection.right()
                y = detection.top(); bottom = detection.bottom()  
                w = right - x
                h = bottom - y  
                all_detections.append([x, y, w, h])
                cv2.rectangle(frame, (x, y), (x+w, y+h),(255,255,255), 1) #highlight detected face
        else:
            detections = self.mtcnn_detector.detect_faces(frame)
            for detection in detections:
                confidence_score = str(round(100*detection["confidence"], 2))+"%"
                x, y, w, h = detection["box"]
                all_detections.append([x, y, w, h])
                cv2.putText(frame, confidence_score, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.rectangle(frame, (x, y), (x+w, y+h),(255,255,255), 1) #highlight detected face

        return frame, all_detections