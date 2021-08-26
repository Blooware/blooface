import cv2
import BlooFace

blooFace = BlooFace.Blooface()

# Get webcam feed
cap = cv2.VideoCapture(0)

resize = .5
font = cv2.FONT_HERSHEY_SIMPLEX

# For each frame
while True:
    # Get frame
    ret, frame = cap.read()

    # Resize
    frame = cv2.resize(frame, None, fx=resize, fy=resize, interpolation=cv2.INTER_AREA)

    # Detect faces
    faces, region = blooFace.query_image(frame)

    if len(faces) > 0:

        print(faces, region)
        # For each face
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(frame, faces[0], (7, 70), font, .5, (100, 255, 0), 1, cv2.LINE_AA)
    # for (x, y, w, h) in regions:
    #     # Draw rectangle around face
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display frame
    cv2.imshow('Blooface', frame)

    # Quit if q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break