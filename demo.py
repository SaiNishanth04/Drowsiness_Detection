import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import pygame
from imutils import face_utils

# Function to calculate the Euclidean distance between two points
def euclidean_distance(ptA, ptB):
    return np.linalg.norm(ptA - ptB)
def eye_aspect_ratio(eye):
    
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    
    C = dist.euclidean(eye[0], eye[3])
  
    ear = (A + B) / (2.0 * C)
    return ear

# Initialize pygame for sound alerts
pygame.mixer.init()
alert_sound = pygame.mixer.Sound('C:\\Users\\Visitor\\Desktop\\Drowsiness\\alarm.wav')  # Replace 'alarm.wav' with your own alert sound file

# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('C:\\Users\\Visitor\\Desktop\\Drowsiness\\shape_predictor_68_face_landmarks.dat')  # Downloaded from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

# Constants for drowsiness detection
EYE_AR_THRESHOLD = 0.25
EYE_AR_CONSEC_FRAMES = 30
YAWN_CONSEC_FRAMES = 10  # Number of consecutive frames to detect a yawn
COUNTER_EYE = 0
COUNTER_YAWN = 0
ALARM_ON = False
YAWN_THRESHOLD = 0.0492
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    if len(faces) > 0:
        face = faces[0]
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        left_eye = shape[42:48]
        right_eye = shape[36:42]

        # Calculate eye aspect ratio (EAR) for drowsiness detection
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        ear = (left_ear + right_ear) / 2.0

        # Calculate the mouth aspect ratio (MAR) for yawn detection
        mouth = shape[48:68]
        mar = euclidean_distance(mouth[2], mouth[10]) / euclidean_distance(mouth[0], mouth[6])

        cv2.drawContours(frame, [cv2.convexHull(left_eye)], -1, (0, 0, 255), 2)
        cv2.drawContours(frame, [cv2.convexHull(right_eye)], -1, (0, 0, 255), 2)
        cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (0, 0, 255), 2)

        
        if ear < EYE_AR_THRESHOLD:
            COUNTER_EYE += 1
            if ALARM_ON and not pygame.mixer.get_busy():
                alert_sound.play()
            if COUNTER_EYE >= EYE_AR_CONSEC_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True
                else:
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER_EYE = 0
            ALARM_ON = False
            pygame.mixer.Sound.stop(alert_sound)

        if mar < YAWN_THRESHOLD:
            COUNTER_YAWN += 1
            if COUNTER_YAWN >= YAWN_CONSEC_FRAMES:
                cv2.putText(frame, "YAWN DETECTED!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER_YAWN = 0

        cv2.putText(frame, f"EAR: {ear:.2f}", (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (500, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
