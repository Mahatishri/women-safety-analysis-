import cv2
import numpy as np
import mediapipe as mp
import os
from ultralytics import YOLO

# Load YOLOv10 model
yolo_model = YOLO('yolov10s.pt')

# Initialize MediaPipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Directories
data_dir = 'ManualyClippedvideo'
classes = [d for d in os.listdir(data_dir)]
print(classes)

# Parameters
max_people = 10  # Maximum number of people to handle in a single frame
num_keypoints = 33  # Number of key points detected by MediaPipe
feature_dim = 3  # x, y, visibility
num_frames = 5  # Number of frames per sequence

sequences = []
labels = []

for label, class_name in enumerate(classes):
    class_dir = os.path.join(data_dir, class_name)
    for video_name in os.listdir(class_dir):
        video_path = os.path.join(class_dir, video_name)
        cap = cv2.VideoCapture(video_path)
        
        frame_counter = 0
        pose_sequences = [[] for _ in range(max_people)]  # Sequence for each detected person
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO model to detect people
            results = yolo_model(frame)

            frame_counter += 1
            print(f'Processing frame {frame_counter}')
            
            for result in results:
                boxes = result.boxes
                people_count = 0
                
                for box in boxes:
                    print(people_count)
                    if people_count >= max_people:
                        break  # Process only up to 'max_people'
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    class_id = int(box.cls)

                    # Ensure we are processing the "person" class (YOLO class_id for person is typically 0)
                    if class_id == 0:
                        # Ensure bounding box is within the image dimensions
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                        # Cropping the person from the frame
                        person_image = frame[y1:y2, x1:x2]

                        if person_image.size > 0:
                            # Convert the cropped image to RGB
                            person_image_rgb = cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB)
                            
                            # Process with MediaPipe Pose
                            results_pose = pose.process(person_image_rgb)

                            if results_pose.pose_landmarks:
                                # Collect keypoints for this person
                                pose_frame = []
                                for lm in results_pose.pose_landmarks.landmark:
                                    pose_frame.extend([lm.x, lm.y, lm.visibility])
                                
                                # Add pose keypoints to the corresponding person's sequence
                                pose_sequences[people_count].append(pose_frame)

                        people_count += 1
            
            # If enough frames have been collected for each person, store the sequences
            for i in range(people_count):
                if len(pose_sequences[i]) == num_frames:
                    sequences.append(np.array(pose_sequences[i]))
                    labels.append(label)
                    pose_sequences[i] = []  # Reset sequence for next set of frames
        
        cap.release()

# Convert to numpy arrays
sequences = np.array(sequences)
labels = np.array(labels)

# Save the data
np.save('sequences1.npy', sequences)
np.save('labels1.npy', labels)
