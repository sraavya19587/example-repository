import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Capture webcam
cap = cv2.VideoCapture(0)

# State tracking
prev_x, prev_y = 0, 0
gesture_cooldown = 1  # seconds
last_gesture_time = 0

def send_key(action):
    global last_gesture_time
    if time.time() - last_gesture_time > gesture_cooldown:
        pyautogui.press(action)
        print(f"Action: {action}")
        last_gesture_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Track index finger tip
        index_finger = hand_landmarks.landmark[8]
        cx, cy = int(index_finger.x * w), int(index_finger.y * h)

        cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)

        dx = cx - prev_x
        dy = cy - prev_y

        # Detect directional movement
        if abs(dx) > 40:
            if dx > 0:
                send_key('right')
            else:
                send_key('left')
        elif dy < -40:
            send_key('up')     # Jump
        elif dy > 40:
            send_key('down')   # Roll

        prev_x, prev_y = cx, cy

    cv2.putText(frame, "Use hand to control: left/right/up/down", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.imshow("Subway Surfer Hand Controller", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
