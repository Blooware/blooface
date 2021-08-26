# Blooface
This is a simple wrapper around the popular DeepFace framework.
https://viso.ai/computer-vision/deepface/

It's main objective is to provide a flexible and lightweight face recognition module for a number of projects.

In the interest of AI and data safety, we have open sourced this module.

To identify faces in a given frame (Numpy array representation):
'''
faces, region = blooFace.query_image(frame)
'''

To verify that two faces are the same:
'''
blooFace.verify(img_path_1, img_path_2)
'''

An example of a simple webcam feed implementation:
'''
import cv2
import BlooFace

blooFace = BlooFace.Blooface()

# Get webcam feed
cap = cv2.VideoCapture(0)

resize = .5
font = cv2.FONT_HERSHEY_SIMPLEX

while True:

    # Get frame
    ret, frame = cap.read()

    # Resize
    frame = cv2.resize(frame, None, fx=resize, fy=resize, interpolation=cv2.INTER_AREA)

    # Detect faces
    faces, region = blooFace.query_image(frame)

    if len(faces) > 0:
        # For each face
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, faces[0], (7, 70), font, .5, (100, 255, 0), 1, cv2.LINE_AA)
   
    # Display frame
    cv2.imshow('Blooface', frame)

    # Quit if q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
'''
-------------------------------------------
FPS Credit:
https://www.geeksforgeeks.org/python-displaying-real-time-fps-at-which-webcam-video-file-is-processed-using-opencv/