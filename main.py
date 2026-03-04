
import cv2
import dlib
import numpy as np
from imutils import face_utils
import math

# Load face detector & predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

def get_head_angle(landmarks):
    nose = landmarks[30]
    chin = landmarks[8]
    dx = nose[0] - chin[0]
    dy = nose[1] - chin[1]
    angle = math.degrees(math.atan2(dy, dx))
    return angle

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) > 1:
        cv2.putText(frame, "ALERT: Multiple Faces Detected!", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        angle = get_head_angle(landmarks)

        if angle > 100 or angle < 80:
            cv2.putText(frame, "ALERT: Looking Away!", (20,80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    cv2.imshow("AI Exam Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
