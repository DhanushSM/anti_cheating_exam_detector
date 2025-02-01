import cv2
import mediapipe as mp
import numpy as np
import datetime

# Initialize Mediapipe face detection, pose estimation, and face mesh
mp_face_detection = mp.solutions.face_detection
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Initialize models
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
pose_estimation = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load YOLO model for mobile phone detection
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")  # Download YOLOv4 weights and cfg
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO class labels
with open("coco (2).names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define the class ID for "cell phone" in COCO dataset
CELL_PHONE_CLASS_ID = 67  # "cell phone" is class 67 in COCO

# Initialize video writer
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 20
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, fps, (frame_width, frame_height))

# Flag to indicate if cheating is detected
cheating_detected = False
last_position = None
frame_not_detected_for = 0  # Time user has been out of the screen
looking_away_count = 0  # Number of chances for looking away
max_looking_away_count = 4  # Max number of chances for looking away
malpractice_score = 0  # Score for malpractice detection


def detect_cheating(frame):
    global cheating_detected, last_position, frame_not_detected_for, looking_away_count, malpractice_score
    # Convert frame to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_results = face_detection.process(rgb_frame)
    if face_results.detections:
        if len(face_results.detections) > 1:
            print("Cheating Detected: Multiple faces in the frame!")
            cv2.putText(frame, "Multiple Faces Detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cheating_detected = True
            malpractice_score += 30  # Increase score for multiple faces

    # Detect pose (head orientation)
    pose_results = pose_estimation.process(rgb_frame)
    if pose_results.pose_landmarks:
        # Check if the user is looking away (e.g., nose landmark position)
        nose = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        if nose.x < 0.2 or nose.x > 0.8 or nose.y < 0.2 or nose.y > 0.8:
            print("Cheating Detected: User is looking away!")
            cv2.putText(frame, "Looking Away Detected!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cheating_detected = True
            looking_away_count += 1
            malpractice_score += 20  # Increase score for looking away

    # Detect mobile phones using YOLO
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process YOLO outputs
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # Filter for cell phone detection
            if class_id == CELL_PHONE_CLASS_ID and confidence > 0.5:
                # Get bounding box coordinates
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-max suppression to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels for mobile phones
    for i in indices:
        box = boxes[i]
        x, y, w, h = box
        label = f"Cell Phone: {confidences[i]:.2f}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        print("Cheating Detected: Mobile phone in the frame!")
        cv2.putText(frame, "Mobile Phone Detected!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cheating_detected = True
        malpractice_score += 40  # Increase score for mobile phone detection

    # Track the user's position (based on face or pose landmarks) for detection
    if pose_results.pose_landmarks:
        current_position = (pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x,
                            pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y)
        if last_position:
            if np.abs(last_position[0] - current_position[0]) > 0.1 or np.abs(
                    last_position[1] - current_position[1]) > 0.1:
                frame_not_detected_for = 0  # Reset counter if user is back on screen
            else:
                frame_not_detected_for += 1
        last_position = current_position

    # If user has been out of the frame for 7 seconds, stop recording
    if frame_not_detected_for > fps * 7:  # More than 7 seconds
        print("Cheating Detected: User out of screen for too long!")
        cv2.putText(frame, "User Out of Screen!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cheating_detected = True
        malpractice_score += 50  # Increase score for being out of screen

    # Display malpractice score
    if malpractice_score >= 90:
        cv2.putText(frame, f"Malpractice Score: {malpractice_score} (High Risk)", (10, 180), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)
    elif malpractice_score >= 95:
        cv2.putText(frame, f"Malpractice Score: {malpractice_score} (Critical Risk)", (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Detect cheating
    frame = detect_cheating(frame)
    # If cheating is detected, take a screenshot and stop recording
    if cheating_detected:
        # Take screenshots as proof
        if looking_away_count <= max_looking_away_count:
            screenshot_path = f"cheating_proof_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(screenshot_path, frame)
            print(f"Screenshot saved as {screenshot_path}")
            looking_away_count += 1
        else:
            break  # Stop recording after 4 screenshots

    # Write the frame to the video file
    out.write(frame)

    # Show the video feed
    cv2.imshow("Malpractice Detection", frame)

    # Stop recording with the "q" key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
