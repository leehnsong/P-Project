import cv2
cap = cv2.VideoCapture("videos/partition1_video.mp4")
print(cap.get(3), cap.get(4))   # width, height
