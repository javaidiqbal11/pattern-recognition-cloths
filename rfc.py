import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


# Function to load images and labels from directory structure
def load_images_from_directory(base_dir):
    images = []
    labels = []
    label_names = os.listdir(base_dir)
    for label in label_names:
        label_dir = os.path.join(base_dir, label)
        if os.path.isdir(label_dir):
            for filename in os.listdir(label_dir):
                img_path = os.path.join(label_dir, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (150, 150))  # Resize to 32x32 pixels
                    img = img.flatten()  # Flatten the image
                    images.append(img)
                    labels.append(label)
    return np.array(images), np.array(labels)


# Load images and labels
base_dir = 'pattern-recognition/train'
images, labels = load_images_from_directory(base_dir)

# Encode labels as integers
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=42)

# Initialize and train the classifier
clf = RandomForestClassifier(n_estimators=500, random_state=42, max_depth=5)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the classifier
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
