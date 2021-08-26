import cv2
import time
import BlooFace

bf = BlooFace.Blooface(train=False)

# Capture webcam feed
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Resize frame
    resize = 1
    frame = cv2.resize(frame, (0, 0), fx=resize, fy=resize)
    
    tic = time.time()
    face, region = bf.detect(frame, False, 'ssd')

    # Draw region bounding box
    if region is not None:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    toc = time.time()

    cv2.putText(frame, 'FPS: {:.2f}'.format(1 / (toc - tic)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break