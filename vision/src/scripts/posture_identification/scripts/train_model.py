from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
import os

# Set up current script directory and load embeddings
current_dir = os.path.dirname(os.path.abspath(__file__))
embeddings_path = os.path.join(current_dir, "..", "output", "pose_embeddings.pickle")

print("[INFO] Loading pose embeddings...")
with open(embeddings_path, "rb") as f:
    data = pickle.load(f)

# Encode labels
print("[INFO] Encoding posture labels...")
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(data["labels"])

# Train a Support Vector Classifier (SVC) on the pose embeddings
print("[INFO] Training posture classification model...")
svm_classifier = SVC(C=1.0, kernel="rbf", probability=True)
svm_classifier.fit(data["embeddings"], encoded_labels)

# Save the trained model and label encoder to disk
recognizer_path = os.path.join(current_dir, "..", "output", "pose_recognizer.pickle")
le_path = os.path.join(current_dir, "..", "output", "label_encoder.pickle")

print(f"[INFO] Saving model to {recognizer_path} and label encoder to {le_path}...")
with open(recognizer_path, "wb") as f:
    pickle.dump(svm_classifier, f)
with open(le_path, "wb") as f:
    pickle.dump(label_encoder, f)
