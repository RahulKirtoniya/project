import cv2
import numpy as np
import os

# Step 1: Load Known Faces
known_faces = []
known_names = []

# Update the path to your dataset folder
train_dir = 'C:/Users/HP/Desktop/New folder/Project Data'

# Ensure valid size for EigenFaces (OpenCV uses 92x112)
target_size = (92, 112)

for filename in os.listdir(train_dir):
    img_path = os.path.join(train_dir, filename)
    img = cv2.imread(img_path)

    if img is None:
        print(f"[Error] Unable to read image file: {img_path}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    try:
        resized_face = cv2.resize(gray, target_size)
    except Exception as e:
        print(f"[Warning] Skipping {filename}: cannot resize. {e}")
        continue

    known_faces.append(resized_face)
    known_names.append(os.path.splitext(filename)[0])  # Filename without extension

# Step 2: Check for empty dataset
if not known_faces:
    print("[Error] No valid training data found.")
    exit()

# Step 3: Train the recognizer
label_ids = {name: idx for idx, name in enumerate(known_names)}
labels = np.array([label_ids[name] for name in known_names])

recognizer = cv2.face.EigenFaceRecognizer_create()
recognizer.train(known_faces, labels)

# Step 4: Initialize camera and classifier
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Attendance tracking
marked_attendance = set()

while True:
    ret, frame = cap.read()
    if not ret:
        print("[Error] Failed to capture frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, target_size)
        
        label, confidence = recognizer.predict(face_resized)

        if confidence < 3000:  # Eigenfaces confidence threshold
            name = list(label_ids.keys())[list(label_ids.values()).index(label)]
            if name not in marked_attendance:
                print(f"[Info] Attendance marked for {name}")
                marked_attendance.add(name)
                # Optional: Save to CSV or database here

            # Draw box and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({int(confidence)})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, "Unknown", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
