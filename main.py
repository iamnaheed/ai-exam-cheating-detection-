import cv2
import dlib
import numpy as np
import joblib
from imutils import face_utils
from utils.head_pose_estimation import calculate_head_angle
from utils.feature_extraction import extract_features

# Load trained model
model = joblib.load("models/cheating_model.pkl")

# Load face detector & landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    face_count = len(faces)

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        angle = calculate_head_angle(landmarks)

        features = extract_features(landmarks, face_count)
        prediction = model.predict([features])[0]

        if prediction == 1:
            cv2.putText(frame, "ALERT: Suspicious Activity!",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Normal",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)

    cv2.imshow("AI Exam Cheating Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
