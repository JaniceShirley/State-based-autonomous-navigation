import cv2
import pyttsx3
import threading
import time

engine = pyttsx3.init()
engine.setProperty('rate', 170)

last_command = ""
last_spoken_time = 0
speech_cooldown = 1.0  # seconds
def speak(text):
    def run():
        engine.stop()
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run, daemon=True).start()
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

# Robot initial position (will be set dynamically)
robot_x = None
robot_y = None
walk_phase = 0
mode = "AUTONOMOUS"
last_move = None
prev_grid = None
smoothing_alpha = 0.7  # higher = more stable (0.6–0.8 good)

escape_timer = 0
escape_duration = 20  # frames to commit after escape

def draw_robot(frame, x, y, phase):
    # Body (rounded rectangle style)
    cv2.rectangle(frame, (x-30, y-60), (x+30, y), (50, 150, 255), -1)
    cv2.rectangle(frame, (x-25, y-65), (x+25, y-60), (50, 150, 255), -1)

    # Head
    cv2.circle(frame, (x, y-90), 30, (255, 255, 0), -1)

    # Eyes
    cv2.circle(frame, (x-10, y-95), 6, (0, 0, 0), -1)
    cv2.circle(frame, (x+10, y-95), 6, (0, 0, 0), -1)

    # Antenna
    cv2.line(frame, (x, y-120), (x, y-140), (0, 255, 0), 3)
    cv2.circle(frame, (x, y-145), 5, (0, 0, 255), -1)

    # Arms
    arm_offset = 8 if phase % 20 < 10 else -8
    cv2.line(frame, (x-30, y-40), (x-50, y-40 + arm_offset), (255, 255, 255), 4)
    cv2.line(frame, (x+30, y-40), (x+50, y-40 - arm_offset), (255, 255, 255), 4)

    # Wheels
    cv2.circle(frame, (x-15, y+10), 12, (0, 0, 0), -1)
    cv2.circle(frame, (x+15, y+10), 12, (0, 0, 0), -1)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    key = cv2.waitKey(1) & 0xFF

    height, width, _ = frame.shape
    # Initialize robot position once at bottom-center
    if robot_x is None and robot_y is None:
        robot_x = width // 2
        robot_y = height - 120
    left_zone = width // 3
    right_zone = 2 * width // 3

    if mode == "AUTONOMOUS":
        results = model(frame)
    else:
        results = []

    if mode == "AUTONOMOUS":
        command = "MOVE FORWARD"

        # --- 3 Vertical Region Navigation ---
        regions = {
            "LEFT": {"total": 0, "bottom": 0},
            "CENTER": {"total": 0, "bottom": 0},
            "RIGHT": {"total": 0, "bottom": 0}
        }

        mid_height = height // 2

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                box_area = (x2 - x1) * (y2 - y1)

                if box_area < 2000:
                    continue

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                if cx < left_zone:
                    region = "LEFT"
                elif cx < right_zone:
                    region = "CENTER"
                else:
                    region = "RIGHT"

                regions[region]["total"] += box_area

                if cy > mid_height:
                    regions[region]["bottom"] += box_area

        # Persistent region memory
        if "current_region" not in globals():
            current_region = None

        # If no region selected yet OR obstacle now in current region → choose new safest
        if current_region is None or regions[current_region]["total"] > 5000:
            current_region = min(regions, key=lambda r: regions[r]["total"])

        # Determine target X (center of chosen region)
        if current_region == "LEFT":
            target_x = left_zone // 2
        elif current_region == "CENTER":
            target_x = (left_zone + right_zone) // 2
        else:
            target_x = (right_zone + width) // 2

        target_y = height - 120  # stay roughly bottom unless vertical avoidance needed

        # Vertical avoidance inside region
        if regions[current_region]["bottom"] > 3000:
            target_y = height // 3  # move upward inside region

        dx = target_x - robot_x
        dy = target_y - robot_y

        threshold = 15

        if abs(dx) < threshold and abs(dy) < threshold:
            command = "STOP"
        else:
            command = "MOVE"
    else:
        command = "MANUAL CONTROL"

    # Update robot position
    if mode == "AUTONOMOUS":
        speed = 0.08  # smooth proportional speed

        robot_x += dx * speed
        robot_y += dy * speed

        robot_x = int(robot_x)
        robot_y = int(robot_y)

    elif mode == "MANUAL":
        if key == ord('l'):   # move left
            robot_x -= 12
        elif key == ord('r'): # move right
            robot_x += 12
        elif key == ord('u'): # move up
            robot_y -= 8
        elif key == ord('d'): # move down
            robot_y += 8

    # Keep robot inside screen bounds
    robot_x = max(70, min(robot_x, width - 70))
    robot_y = max(120, min(robot_y, height - 70))

    walk_phase += 3

    # Non-blocking speech with cooldown
    current_time = time.time()

    if command != last_command and (current_time - last_spoken_time) > speech_cooldown:
        if command == "STOP":
            speak("Stopping")
        elif command == "MOVE LEFT":
            speak("Moving left")
        elif command == "MOVE RIGHT":
            speak("Moving right")
        elif command == "MOVE FORWARD":
            speak("Moving forward")

        last_command = command
        last_spoken_time = current_time

    annotated_frame = frame.copy()

    # Draw semi-transparent dark overlay
    overlay = annotated_frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 100), (30, 30, 30), -1)
    alpha = 0.6
    annotated_frame = cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0)

    # Draw navigation zones
    cv2.line(annotated_frame, (left_zone, 0), (left_zone, height), (255, 255, 255), 2)
    cv2.line(annotated_frame, (right_zone, 0), (right_zone, height), (255, 255, 255), 2)

    # Status Panel (Top Left)
    cv2.putText(annotated_frame, f"{mode} MODE", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)

    cv2.putText(annotated_frame, f"COMMAND: {command}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Draw robot
    draw_robot(annotated_frame, robot_x, robot_y, walk_phase)

    # Draw custom clean bounding boxes
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Draw clean bounding box
            color = (0, 255, 0)
            if label == "person":
                color = (0, 0, 255)

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Robot Navigation System", annotated_frame)

    # Mode switching
    if key == ord('1'):
        mode = "AUTONOMOUS"
    elif key == ord('2'):
        mode = "MANUAL"

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()