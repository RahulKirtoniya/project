import face_recognition
import cv2
import numpy as np
import pandas as pd
from datetime import date

# Step 1: Load Known Faces
def load_known_faces(dataset_path):
    known_faces = []
    known_face_names = []

    for filename in os.listdir(dataset_path):
        image_path = os.path.join(dataset_path, filename)
        image = face_recognition.load_image_file(image_path)
        face_encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(face_encoding)
        known_face_names.append(filename.split('.')[0])  # Use filename without extension as face name

    return known_faces, known_face_names

# Step 2: Detect and Recognize Faces
def detect_and_recognize_faces(video_stream, known_faces, known_face_names):
    attendance_record = {}

    while True:
        ret, frame = video_stream.read()
        rgb_frame = frame[:, :, ::-1]  # Convert BGR to RGB

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_faces, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_faces, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                attendance_record[name] = 'Present'

            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (255, 0, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        cv2.imshow('Face Detection and Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_stream.release()
    cv2.destroyAllWindows()
    return attendance_record

# Step 3: Save Attendance
def save_attendance(attendance_record):
    today = date.today().isoformat()
    df = pd.DataFrame(attendance_record.items(), columns=['Student Name', 'Attendance'])
    df.to_csv(f'attendance_{today}.csv', index=False)
    print(f"Attendance saved to attendance_{today}.csv")

# Step 4: Main Function
if __name__ == "__main__":
    dataset_path = 'path_to_your_dataset'  # Set the path to your dataset
    known_faces, known_face_names = load_known_faces(dataset_path)

    video_stream = cv2.VideoCapture(0)  # Use the default camera
    attendance_record = detect_and_recognize_faces(video_stream, known_faces, known_face_names)
    save_attendance(attendance_record)