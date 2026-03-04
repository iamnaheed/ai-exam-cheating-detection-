def extract_features(landmarks, face_count):
    nose_chin_angle = abs(landmarks[30][1] - landmarks[8][1])
    eye_left_x = landmarks[36][0]
    eye_right_x = landmarks[45][0]

    return [
        nose_chin_angle,
        face_count,
        eye_left_x,
        eye_right_x
    ]
