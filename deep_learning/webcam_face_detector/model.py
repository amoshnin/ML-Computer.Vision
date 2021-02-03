import argparse
import cv2
from lib_detector.face_detector import FaceDetector

ap = argparse.ArgumentParser()
ap.add_argument('-f', '--face', required=False, help='Path to where the face cascade resides', default='./cascades/haarcascade_frontalface_default.xml')
ap.add_argument('-v', '--video', required=False, help='Path to where the video file resides')
args = vars(ap.parse_args())

fd = FaceDetector(args['face'])

if not args.get("video", False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args['video'])

