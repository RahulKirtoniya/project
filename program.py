import cv2
import numpy as np
import os

# Load the known faces and their corresponding names
known_faces = []
known_names = []

# Load the training data (e.g., images of known individuals)
train_dir = 'C:/Users/HP/Desktop/New folder/Project Data'
for filename in os.listdir(train_dir):
    img_path = os.path.join(train_dir, filename)
    img = cv2.imread(img_path)
    
    # Check if the image file can be read
    if img is None:
        print(f"Error: Unable to read image file {img_path}")
        continue
    
    # Check if the image file is in a valid format
    if not img.shape == (112, 92):
        print(f"Error: Image file {img_path} is not in a valid format")
        continue
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_roi = gray  # Extract the face region
    known_faces.append(face_roi.reshape(-1, 1))  # Reshape for Eigenfaces input
    known_names.append(filename.split('.')[0])  # Extract the name from the filename

# Check if the training data is empty
if len(known_faces) == 0:
    print("Error: No training data available")
    exit()

# Create a dictionary to map names to integers
name_to_label = {name: i for i, name in enumerate(known_names)}

# Create an Eigenfaces recognizer
recognizer = cv2.face.EigenFaceRecognizer_create()

# Train the Eigenfaces recognizer with the known faces
labels = np.array([name_to_label[name] for name in known_names])
recognizer.train(known_faces, labels)

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()
    
    # Detect faces in the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    # Loop through each detected face
    for (x, y, w, h) in faces:
        # Extract the facial features
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (92, 112))  # Resize to Eigenfaces input size
        face_roi = face_roi.reshape(-1, 1)  # Reshape for Eigenfaces input
        
        # Recognize the face
        label, confidence = recognizer.predict(face_roi)
        
        # If a match is found, mark attendance
        if confidence < 50:  # Adjust the confidence threshold as needed
            name = known_names[label]
            print(f"Attendance marked for {name}")
            # Store the attendance data in the database
            # ...
    
    # Display the output
    cv2.imshow('Attendance System', frame)
    
    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()