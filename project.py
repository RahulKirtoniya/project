import face_recognition
import cv2
import numpy as np
import pandas as pd
from datetime import date
import os

# Step 1: Load Known Faces
def load_known_faces(dataset_path):
    known_faces = []
    known_face_names = []

    for filename in os.listdir(dataset_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Ensure valid image files
            image_path = os.path.join(dataset_path, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            if encodings:  # Check if face is detected
                known_faces.append(encodings[0])
                known_face_names.append(os.path.splitext(filename)[0])  # Remove extension
            else:
                print(f"[Warning] No face found in {filename}")

    return known_faces, known_face_names

# Step 2: Detect and Recognize Faces
def detect_and_recognize_faces(video_stream, known_faces, known_face_names):
    attendance_record = {}

    while True:
        ret, frame = video_stream.read()
        if not ret:
            print("[Error] Failed to grab frame.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_faces, face_encoding)
            face_distances = face_recognition.face_distance(known_faces, face_encoding)
            best_match_index = np.argmin(face_distances)
            name = "Unknown"

            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                attendance_record[name] = 'Present'

            # Draw rectangle and label
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)

        cv2.imshow('Face Detection and Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_stream.release()
    cv2.destroyAllWindows()
    return attendance_record

# Step 3: Save Attendance
def save_attendance(attendance_record):
    today = date.today().isoformat()
    df = pd.DataFrame(attendance_record.items(), columns=['Name', 'Attendance'])
    filename = f'attendance_{today}.csv'
    df.to_csv(filename, index=False)
    print(f"[Info] Attendance saved to {filename}")

# Step 4: Main Function
if __name__ == "__main__":
    dataset_path = 'path_to_your_dataset'  # Replace with the actual path to your dataset
    known_faces, known_face_names = load_known_faces(dataset_path)

    if known_faces:
        video_stream = cv2.VideoCapture(0)
        attendance_record = detect_and_recognize_faces(video_stream, known_faces, known_face_names)
        save_attendance(attendance_record)
    else:
        print("[Error] No known faces loaded. Please check your dataset path.")
