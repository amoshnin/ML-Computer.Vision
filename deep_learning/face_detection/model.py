import argparse
import cv2
from lib_detector.face_detector import FaceDetector

ap = argparse.ArgumentParser()
ap.add_argument('-f', '--face', required=False, help='Path to where the face cascade resides', default='./cascades/haarcascade_frontalface_default.xml')
ap.add_argument('-i', '--image', required=False, help='Path to where the image file resides' ,default='./images/messi2.png')
args = vars(ap.parse_args())

print(args)
image = cv2.imread(args['image'])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

fd = FaceDetector(args['face'])
faceRects = fd.detect(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
print("I found {} faces".format(len(faceRects)))
print(faceRects)

for (x, y, width, height) in faceRects:
    cv2.rectangle(image, (x, y), (x + width, y+height), (0, 255, 0), 2)

cv2.imwrite('res.png', image)
cv2.imshow('Faces', image)
cv2.waitKey(0)