import cv2

img = cv2.VideoCapture("http://192.168.1.106:4747/video")

while True:
    ret, frame = img.read()
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break