
import math

def calculate_head_angle(landmarks):
    nose = landmarks[30]
    chin = landmarks[8]

    dx = nose[0] - chin[0]
    dy = nose[1] - chin[1]

    angle = math.degrees(math.atan2(dy, dx))
    return abs(angle)
