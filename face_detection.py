import cv2

# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Defining a function that will do the detections
def detect(colored_image, gray):
    face = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in face:
        cv2.rectangle(colored_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = colored_image[y:y+h, x:x+w]
        eye = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for (x_, y_, w_, h_) in eye:
            cv2.rectangle(roi_color, (x_, y_), (x_+w_, y_+h_), (0, 0, 255), 2)
    return colored_image

# Doing some Face Recognition with the webcam
cap = cv2.VideoCapture(0)
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(frame, gray)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()