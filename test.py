import BlooFace


# bf = BlooFace.Blooface(train=False)
# res = bf.query_image("dataset/test.jpg", 3)
# print(res)

import cv2
bf = BlooFace.Blooface(train=False)
res = bf.detect("dataset/huw.jpg")

cv2.imshow("test", res)
cv2.waitKey(0)
