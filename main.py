import cv2
import os
import csv
import numpy as np
from datetime import datetime
import time

class SimpleAttendanceSystem:
    def __init__(self):
        # Set up directories
        self.students_folder = "data/students"
        self.attendance_folder = "attendance_records"
        
        # Ensure directories exist
        os.makedirs(self.students_folder, exist_ok=True)
        os.makedirs(self.attendance_folder, exist_ok=True)
        
        # Load face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Load student images and create face recognizer
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.student_ids = {}
        self.load_student_images()
        
        # Track attendance to avoid duplicates
        self.attendance_marked = set()
        
        # Initialize today's attendance file
        self.today = datetime.now().strftime("%Y-%m-%d")
        self.attendance_file = os.path.join(self.attendance_folder, f"attendance_{self.today}.csv")
        self.initialize_attendance_file()
    
    def load_student_images(self):
        """Load student images and train the face recognizer."""
        print("Loading student images...")
        
        faces = []
        ids = []
        id_counter = 0
        
        for filename in os.listdir(self.students_folder):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                student_name = os.path.splitext(filename)[0]
                image_path = os.path.join(self.students_folder, filename)
                
                # Map student name to ID
                if student_name not in self.student_ids:
                    id_counter += 1
                    self.student_ids[id_counter] = student_name
                
                # Load image and convert to grayscale
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Error loading image: {image_path}")
                    continue
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Detect faces in the image
                detected_faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w, h) in detected_faces:
                    faces.append(gray[y:y+h, x:x+w])
                    ids.append(id_counter)
                    print(f"Loaded face for {student_name}")
        
        # Reverse the student_ids mapping for easy lookup
        self.name_to_id = {name: id for id, name in self.student_ids.items()}
        
        # If faces were found, train the recognizer
        if faces:
            print(f"Training recognizer with {len(faces)} faces...")
            self.recognizer.train(faces, np.array(ids))
            print("Training complete!")
        else:
            print("No faces found in student images.")
    
    def initialize_attendance_file(self):
        """Create or check attendance file for today."""
        if not os.path.exists(self.attendance_file):
            with open(self.attendance_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Name', 'Time'])
    
    def mark_attendance(self, student_name):
        """Mark attendance for a student."""
        if student_name != "Unknown" and student_name not in self.attendance_marked:
            current_time = datetime.now().strftime("%H:%M:%S")
            with open(self.attendance_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([student_name, current_time])
            self.attendance_marked.add(student_name)
            print(f"Marked attendance for {student_name} at {current_time}")
    
    def get_student_name(self, student_id):
        """Get student name from ID."""
        return self.student_ids.get(student_id, "Unknown")
    
    def try_open_webcam(self, max_attempts=3):
        """Try to open webcam with multiple attempts and camera indices."""
        for attempt in range(max_attempts):
            for camera_index in range(3):  # Try camera indices 0, 1, 2
                print(f"Attempting to open camera index {camera_index}...")
                video_capture = cv2.VideoCapture(camera_index)
                
                if video_capture.isOpened():
                    print(f"Successfully opened camera at index {camera_index}")
                    return video_capture
                
                video_capture.release()
                time.sleep(1)  # Wait before trying again
        
        return None
    
    def run(self):
        """Run the attendance system."""
        # Try to open webcam
        video_capture = self.try_open_webcam()
        
        if video_capture is None:
            print("\nError: Could not open any webcam. Please check:")
            print("1. Your webcam is properly connected")
            print("2. It's not being used by another application")
            print("3. You have granted camera permissions to this application")
            print("\nYou can also try the offline mode to test with images:")
            return self.run_offline_mode()
        
        print("Starting attendance system. Press 'q' to quit.")
        
        while True:
            # Capture frame from webcam
            ret, frame = video_capture.read()
            if not ret:
                print("Error: Can't receive frame")
                break
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                # Draw rectangle around the face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Recognize the face
                roi_gray = gray[y:y+h, x:x+w]
                
                try:
                    # Predict the student ID
                    student_id, confidence = self.recognizer.predict(roi_gray)
                    confidence = 100 - int(confidence)
                    
                    # If confidence is high enough, mark attendance
                    if confidence > 50:
                        student_name = self.get_student_name(student_id)
                        self.mark_attendance(student_name)
                        display_text = f"{student_name} ({confidence}%)"
                    else:
                        display_text = f"Unknown ({confidence}%)"
                    
                    # Display name and confidence
                    cv2.putText(frame, display_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                except Exception as e:
                    # Handle potential errors in recognition
                    print(f"Recognition error: {e}")
                    cv2.putText(frame, "Face detected", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('Attendance System', frame)
            
            # Check for exit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release the webcam and close windows
        video_capture.release()
        cv2.destroyAllWindows()
    
    def run_offline_mode(self):
        """Run the system in offline mode using saved images."""
        test_folder = "data/test_images"
        os.makedirs(test_folder, exist_ok=True)
        
        if not os.listdir(test_folder):
            print(f"\nNo test images found in {test_folder}")
            print("Please add some test images to this folder and run the program again.")
            print("Example file path: data/test_images/classroom1.jpg")
            return
        
        print(f"\nRunning in offline mode with images from {test_folder}")
        
        for filename in os.listdir(test_folder):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(test_folder, filename)
                print(f"\nProcessing image: {filename}")
                
                # Read the image
                frame = cv2.imread(image_path)
                if frame is None:
                    print(f"Error loading image: {image_path}")
                    continue
                
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                print(f"Detected {len(faces)} faces")
                
                for (x, y, w, h) in faces:
                    # Draw rectangle around the face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Recognize the face
                    roi_gray = gray[y:y+h, x:x+w]
                    
                    try:
                        # Predict the student ID
                        student_id, confidence = self.recognizer.predict(roi_gray)
                        confidence = 100 - int(confidence)
                        
                        # If confidence is high enough, mark attendance
                        if confidence > 50:
                            student_name = self.get_student_name(student_id)
                            self.mark_attendance(student_name)
                            display_text = f"{student_name} ({confidence}%)"
                        else:
                            display_text = f"Unknown ({confidence}%)"
                        
                        # Display name and confidence
                        cv2.putText(frame, display_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    except Exception as e:
                        # Handle potential errors in recognition
                        print(f"Recognition error: {e}")
                        cv2.putText(frame, "Face detected", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Display the image with results
                cv2.imshow(f'Image: {filename}', frame)
                cv2.waitKey(0)  # Wait for key press
        
        cv2.destroyAllWindows()
        print("\nOffline processing complete!")

def setup_instructions():
    """Print setup instructions for first-time usage."""
    print("\n---- SETUP INSTRUCTIONS ----")
    print("1. Before running this system, install required packages:")
    print("   pip install opencv-contrib-python numpy")
    print("\n2. Add student images to the 'data/students/' folder:")
    print("   - Each image should contain only one clear face")
    print("   - Name the file with the student's name (e.g., john_smith.jpg)")
    print("\n3. For offline testing, add images to 'data/test_images/' folder")
    print("\n4. Run the program again after adding images")
    print("-----------------------------\n")

def test_webcam():
    """Test if webcam is accessible."""
    print("Testing webcam access...")
    for i in range(3):  # Try the first 3 camera indices
        print(f"Trying camera index {i}...")
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Camera {i} works! Captured frame with shape {frame.shape}")
                cap.release()
                return True
            cap.release()
    print("No working webcam found.")
    return False

if __name__ == "__main__":
    # First, test webcam access
    webcam_works = test_webcam()
    
    # Check if the students folder exists and has images
    students_folder = "data/students"
    if not os.path.exists(students_folder) or not os.listdir(students_folder):
        os.makedirs(students_folder, exist_ok=True)
        setup_instructions()
    else:
        print("Starting attendance system...")
        attendance_system = SimpleAttendanceSystem()
        
        if webcam_works:
            attendance_system.run()
        else:
            print("\nRunning in offline mode since webcam is not available.")
            attendance_system.run_offline_mode()