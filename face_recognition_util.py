import face_recognition
import cv2
import os
import numpy as np

class FaceRecognizer:
    def __init__(self, students_folder):
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_student_images(students_folder)
        
    def load_student_images(self, students_folder):
        """Load student images from the given folder and encode their faces."""
        print("Loading student images...")
        for filename in os.listdir(students_folder):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(students_folder, filename)
                student_name = os.path.splitext(filename)[0]
                
                # Load the image and encode the face
                image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(image)
                
                if len(face_encodings) > 0:
                    self.known_face_encodings.append(face_encodings[0])
                    self.known_face_names.append(student_name)
                    print(f"Loaded {student_name}")
                else:
                    print(f"No face found in {filename}")
        
        print(f"Loaded {len(self.known_face_names)} student faces")
    
    def identify_faces(self, frame):
        """Identify faces in the given frame and return their names."""
        # Convert the image from BGR color (OpenCV) to RGB color
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find all the faces in the current frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        face_names = []
        for face_encoding in face_encodings:
            # Compare the face with known faces
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            
            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
            
            face_names.append(name)
        
        return face_locations, face_names