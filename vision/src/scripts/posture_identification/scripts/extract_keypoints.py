import os
import mediapipe as mp
import numpy as np
from imutils import paths
import pickle

# Set up current script directory and model path
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "..", "models", 'pose_landmarker_heavy.task')

# Initialize Mediapipe Pose Landmarker
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

pose_landmarker_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE
)
landmarker = PoseLandmarker.create_from_options(pose_landmarker_options)

# Prepare image paths and data structures for storing embeddings and labels
print("Extracting Pose Landmarks...")
image_paths = list(paths.list_images(os.path.join(current_dir, "..", "dataset")))
embeddings_list = []
labels_list = []
total_processed = 0

# Process each image in the dataset
for idx, image_path in enumerate(image_paths):
    if idx % 50 == 0:
        print(f"Processing image {idx}/{len(image_paths)}")

    # Extract posture category (e.g., Standing, Squatting)
    posture_label = image_path.split(os.path.sep)[-2]

    # Load image and predict pose landmarks
    image = mp.Image.create_from_file(image_path)
    pose_landmarks_result = landmarker.detect(image)
    pose_landmarks_list = pose_landmarks_result.pose_landmarks

    if pose_landmarks_list:
        # Flatten and store landmarks with corresponding label
        pose_landmarks = pose_landmarks_list[0]
        landmarks_array = np.array([[landmark.x, landmark.y] for landmark in pose_landmarks])
        embeddings_list.append(landmarks_array.flatten())
        labels_list.append(posture_label)
        total_processed += 1

# Save embeddings and labels to disk
output_data = {"embeddings": embeddings_list, "labels": labels_list}
output_file_path = os.path.join(current_dir, "..", "output", "pose_embeddings.pickle")

print(f"[INFO] Serializing {total_processed} embeddings to {output_file_path}...")
with open(output_file_path, "wb") as f:
    pickle.dump(output_data, f)
