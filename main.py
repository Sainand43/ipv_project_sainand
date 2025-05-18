import cv2
import os
import csv
import numpy as np
from datetime import datetime
import time
import re
import dlib
import face_recognition
from collections import defaultdict
from sklearn.decomposition import PCA
import pickle
import pandas as pd
import smtplib
from email.message import EmailMessage

#print(dir(cv2.face))
# Check for required libraries
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    print("Warning: face_recognition library not available.")
    print("Deep learning face recognition will be disabled.")
    print("To install: pip install face_recognition")
    FACE_RECOGNITION_AVAILABLE = False

# Check for dlib
try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    print("Warning: dlib library not available.")
    print("Facial landmarks will be disabled.")
    print("To install: pip install dlib")
    DLIB_AVAILABLE = False

# Email configuration (set your SMTP server and sender email/password here)
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
SENDER_EMAIL = 'kundaikarsainand@gmail.com'
SENDER_PASSWORD = 'vsuyvrbnxvktlgmu'  # Use an appvsuyassword if using Gmail

def csv_to_excel(csv_path, excel_path):
    df = pd.read_csv(csv_path)
    df.to_excel(excel_path, index=False)

def send_email_with_attachment(to_email, subject, body, attachment_path):
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = SENDER_EMAIL
    msg['To'] = to_email
    msg.set_content(body)
    # Attach the file
    with open(attachment_path, 'rb') as f:
        file_data = f.read()
        file_name = os.path.basename(attachment_path)
    msg.add_attachment(file_data, maintype='application', subtype='octet-stream', filename=file_name)
    # Send the email
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        print(f"Sent attendance to {to_email}")

class EnhancedAttendanceSystem:
    def __init__(self):
        # Initialize attributes
        self.landmarks_available = False  # Initialize to False to avoid AttributeError
        
        # Set up directories
        self.students_folder = "data/students"
        self.attendance_folder = "attendance_records"
        self.model_folder = "models"
        
        # Ensure directories exist
        os.makedirs(self.students_folder, exist_ok=True)
        os.makedirs(self.attendance_folder, exist_ok=True)
        os.makedirs(self.model_folder, exist_ok=True)
        
        # Load face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Initialize background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
        
        # Initialize CLAHE for histogram equalization
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # Fix: Initialize clahe
        
        # Set up face recognition models
        self.setup_recognizers()
        # Set default recognizer to LBPH for backward compatibility
        self.recognizer = self.lbph_recognizer
        
        # Try to set up deep learning recognition
        try:
            self.setup_deep_learning_recognition()
            self.deep_learning_available = True
        except Exception as e:
            print(f"Deep learning recognition not available: {e}")
            self.deep_learning_available = False
        
        # Try to set up dlib facial landmarks
        try:
            self.setup_facial_landmarks()
            self.landmarks_available = True
        except Exception as e:
            print(f"Facial landmarks not available: {e}")
            self.landmarks_available = False
        
        # Set up trackers
        self.trackers = {}
        self.tracked_faces = {}
        
        # Track attendance to avoid duplicates
        self.attendance_marked = set()
        
        # Initialize today's attendance file
        self.today = datetime.now().strftime("%Y-%m-%d")
        self.attendance_file = os.path.join(self.attendance_folder, f"attendance_{self.today}.csv")
        self.initialize_attendance_file()
        
        # Set up confidence thresholds
        self.recognition_threshold = 60  # Minimum confidence to consider a match
        self.frames_to_average = 10      # Number of frames to average confidence over

    def align_face(self, face_img, gray_frame=None, x=0, y=0, w=0, h=0):
        """Align face based on eye positions for better recognition."""
        if gray_frame is not None and self.landmarks_available:
            # Use dlib for better landmarks if available
            try:
                # Convert to dlib rectangle
                rect = dlib.rectangle(x, y, x+w, y+h)
                
                # Get facial landmarks
                landmarks = self.predictor(gray_frame, rect)
                
                # Get eye centers
                left_eye_center = np.mean([(landmarks.part(36).x, landmarks.part(36).y),
                                         (landmarks.part(37).x, landmarks.part(37).y),
                                         (landmarks.part(38).x, landmarks.part(38).y),
                                         (landmarks.part(39).x, landmarks.part(39).y),
                                         (landmarks.part(40).x, landmarks.part(40).y),
                                         (landmarks.part(41).x, landmarks.part(41).y)], axis=0)
                
                right_eye_center = np.mean([(landmarks.part(42).x, landmarks.part(42).y),
                                          (landmarks.part(43).x, landmarks.part(43).y),
                                          (landmarks.part(44).x, landmarks.part(44).y),
                                          (landmarks.part(45).x, landmarks.part(45).y),
                                          (landmarks.part(46).x, landmarks.part(46).y),
                                          (landmarks.part(47).x, landmarks.part(47).y)], axis=0)
                
                # Calculate angle for alignment
                dy = right_eye_center[1] - left_eye_center[1]
                dx = right_eye_center[0] - left_eye_center[0]
                angle = np.degrees(np.arctan2(dy, dx))
                
                # Get center of face for rotation matrix
                center = ((left_eye_center[0] + right_eye_center[0]) // 2, 
                         (left_eye_center[1] + right_eye_center[1]) // 2)
                
                # Adjust center coordinates relative to the face ROI
                center = (int(center[0] - x), int(center[1] - y))  # Fix: Ensure center is a tuple of integers
                
                # Rotate image to align eyes horizontally
                M = cv2.getRotationMatrix2D(center, angle, 1)
                aligned_face = cv2.warpAffine(face_img, M, (face_img.shape[1], face_img.shape[0]))
                
                return aligned_face
            except Exception as e:
                print(f"Error in face alignment using landmarks: {e}")
                pass
        
        # Fall back to haar cascade for eye detection
        try:
            eyes = self.eye_cascade.detectMultiScale(face_img)
            
            if len(eyes) >= 2:
                # Sort eyes by x-coordinate to get left and right
                eyes = sorted(eyes, key=lambda x: x[0])
                left_eye, right_eye = eyes[:2]
                
                # Calculate center of each eye
                left_eye_center = (left_eye[0] + left_eye[2]//2, left_eye[1] + left_eye[3]//2)
                right_eye_center = (right_eye[0] + right_eye[2]//2, right_eye[1] + right_eye[3]//2)
                
                # Calculate angle between eye centers
                dy = right_eye_center[1] - left_eye_center[1]
                dx = right_eye_center[0] - left_eye_center[0]
                angle = np.degrees(np.arctan2(dy, dx))
                
                # Rotate image to align eyes horizontally
                center = ((left_eye_center[0] + right_eye_center[0]) // 2, 
                         (left_eye_center[1] + right_eye_center[1]) // 2)
                center = (int(center[0]), int(center[1]))  # Fix: Ensure center is a tuple of integers
                M = cv2.getRotationMatrix2D(center, angle, 1)
                aligned_face = cv2.warpAffine(face_img, M, (face_img.shape[1], face_img.shape[0]))
                
                return aligned_face
        except Exception as e:
            print(f"Error in face alignment using haar cascade: {e}")
        
        return face_img  # Return original if alignment fails
        
    def setup_facial_landmarks(self):
        """Set up dlib's facial landmark detector."""
        if not DLIB_AVAILABLE:
            raise ImportError("dlib library is not installed")
        # Check if landmark predictor file exists, download if not
        landmark_path = os.path.join(self.model_folder, "shape_predictor_68_face_landmarks.dat")
        if not os.path.exists(landmark_path):
            print("Facial landmark predictor not found. Please download from:")
            print("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            print(f"Extract and place in {landmark_path}")
            raise FileNotFoundError("Landmark predictor file not found")
        
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(landmark_path)

    def setup_recognizers(self):
        """Set up multiple face recognizers for ensemble approach."""
        # LBPH recognizer
        self.lbph_recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Eigenfaces recognizer
        self.eigen_recognizer = cv2.face.EigenFaceRecognizer_create()
        
        # Fisherfaces recognizer
        self.fisher_recognizer = cv2.face.FisherFaceRecognizer_create()
        
        # Load student images and create face recognizer
        self.student_ids = {}
        self.student_name_map = {}  # Maps student IDs to actual names (without numbers)
        self.load_student_images()

    def setup_deep_learning_recognition(self):
        """Set up face_recognition library for deep learning-based recognition."""
        if not FACE_RECOGNITION_AVAILABLE:
            raise ImportError("face_recognition library is not installed")

        encodings_file = os.path.join(self.model_folder, "face_encodings.pkl")

        # Check if encodings file exists and is up-to-date
        if os.path.exists(encodings_file):
            # Check if the number of images in the students folder matches the saved encodings
            student_images = [f for f in os.listdir(self.students_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
            with open(encodings_file, "rb") as f:
                data = pickle.load(f)
                if len(data["names"]) == len(student_images):
                    print("Loading face encodings from file...")
                    self.known_face_encodings = data["encodings"]
                    self.known_face_names = data["names"]
                    print(f"Loaded {len(self.known_face_names)} face encodings.")
                    return

        print("Generating face encodings...")
        self.known_face_encodings = []
        self.known_face_names = []

        # Load student images and encode faces
        for filename in os.listdir(self.students_folder):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                # Extract student name
                student_name = self.extract_student_name(filename)
                image_path = os.path.join(self.students_folder, filename)

                # Load image and get face encodings
                try:
                    image = face_recognition.load_image_file(image_path)
                    face_locations = face_recognition.face_locations(image)

                    if face_locations:
                        # Get encoding of first face found
                        face_encoding = face_recognition.face_encodings(image, face_locations)[0]

                        # Add to known faces
                        self.known_face_encodings.append(face_encoding)
                        self.known_face_names.append(student_name)
                        print(f"Added deep learning encoding for {student_name}")
                except Exception as e:
                    print(f"Error processing {image_path} for deep learning: {e}")

        # Save encodings to file
        with open(encodings_file, "wb") as f:
            data = {"encodings": self.known_face_encodings, "names": self.known_face_names}
            pickle.dump(data, f)
        print(f"Saved {len(self.known_face_names)} face encodings to file.")

    def extract_student_name(self, filename):
        """Extract the base student name without numbers."""
        # Remove file extension
        name_without_ext = os.path.splitext(filename)[0]
        
        # Extract the name part (remove trailing numbers)
        # This regex matches a string that ends with one or more digits
        match = re.match(r'([a-zA-Z_]+)(\d+)$', name_without_ext)
        if match:
            return match.group(1)  # Return the name part only
        
        return name_without_ext  # Return original name if no numbers found
    
    def load_student_images(self):
        """Load student images and train the face recognizer."""
        print("Loading student images...")
        
        faces = []
        face_sizes = []
        ids = []
        id_counter = 0
        student_names_seen = {}  # Track which student names we've seen
        
        for filename in os.listdir(self.students_folder):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                # Extract the student name without the number suffix
                student_name_with_num = os.path.splitext(filename)[0]
                student_name = self.extract_student_name(filename)
                
                image_path = os.path.join(self.students_folder, filename)
                
                # Create or get the student ID
                if student_name not in student_names_seen:
                    id_counter += 1
                    student_names_seen[student_name] = id_counter
                    self.student_ids[id_counter] = student_name
                    print(f"Registering new student: {student_name}")
                
                student_id = student_names_seen[student_name]
                
                # Load image and convert to grayscale
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Error loading image: {image_path}")
                    continue
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Detect faces in the image
                detected_faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w, h) in detected_faces:
                    face_roi = gray[y:y+h, x:x+w]
                    
                    # Apply preprocessing
                    quality_ok, quality_msg = self.assess_face_quality(face_roi)
                    if not quality_ok:
                        print(f"Skipping face from {student_name_with_num}: {quality_msg}")
                        continue
                    
                    # Apply face alignment
                    aligned_face = self.align_face(face_roi, gray, x, y, w, h)
                    
                    # Apply lighting normalization
                    normalized_face = self.normalize_lighting(aligned_face)
                    
                    # Store processed face for training
                    faces.append(normalized_face)
                    ids.append(student_id)
                    face_sizes.append((normalized_face.shape[0], normalized_face.shape[1]))
                    print(f"Loaded face from {student_name_with_num} (ID: {student_id})")
        
        # If faces were found, train the recognizers
        if faces:
            print(f"Training recognizers with {len(faces)} faces from {len(student_names_seen)} students...")
            
            # Train LBPH recognizer
            self.lbph_recognizer.train(faces, np.array(ids))
            print("LBPH training complete!")
            
            # For Eigenfaces and Fisherfaces, we need to resize all faces to the same dimensions
            if len(face_sizes) > 0:
                # Find minimum face size to resize all faces to the same dimensions
                min_height = min(face_sizes, key=lambda x: x[0])[0]
                min_width = min(face_sizes, key=lambda x: x[1])[1]
                
                # Resize faces for Eigenfaces and Fisherfaces
                resized_faces = [cv2.resize(f, (min_width, min_height)) for f in faces]
                
                try:
                    # Train Eigenfaces recognizer
                    self.eigen_recognizer.train(resized_faces, np.array(ids))
                    print("Eigenfaces training complete!")
                    
                    # Train Fisherfaces recognizer
                    self.fisher_recognizer.train(resized_faces, np.array(ids))
                    print("Fisherfaces training complete!")
                except Exception as e:
                    print(f"Error training Eigenfaces/Fisherfaces: {e}")
                    print("Falling back to LBPH only")
        else:
            print("No faces found in student images.")
    
    def assess_face_quality(self, face_img):
        """Assess if a face image is good enough for recognition."""
        # Check brightness
        brightness = np.mean(face_img)
        if brightness < 40:
            return False, "Too dark"
        if brightness > 240:
            return False, "Too bright"
        
        # Check face size
        h, w = face_img.shape
        if h < 100 or w < 100:
            return False, "Face too small"
        
        # Check blur using Laplacian variance
        laplacian = cv2.Laplacian(face_img, cv2.CV_64F).var()
        if laplacian < 100:
            return False, "Image too blurry"
        
        return True, "Good quality"
    
    def normalize_lighting(self, face_img):
        """Apply histogram equalization to normalize lighting conditions."""
        return self.clahe.apply(face_img)
    
    def initialize_attendance_file(self):
        """Create or check attendance file for today."""
        if not os.path.exists(self.attendance_file):
            with open(self.attendance_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Name', 'Time', 'Confidence', 'Method'])
    
    def mark_attendance(self, student_name, confidence=0, method="Unknown"):
        """Mark attendance for a student based on the current subject, only once per lecture."""
        # Get the current time
        current_time = datetime.now().strftime("%H:%M")
        today_date = datetime.now().strftime("%Y-%m-%d")
        # Load the timetable
        timetable_file = "timetable.csv"
        if not os.path.exists(timetable_file):
            print("Error: Timetable file not found.")
            return
        with open(timetable_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                start_time = row['start_time']
                end_time = row['end_time']
                subject = row['subject']
                # Check if the current time is within the subject's time range
                if start_time <= current_time <= end_time:
                    # Save as Subjectname_Date_attendance.csv
                    subject_attendance_file = os.path.join(self.attendance_folder, f"{subject}_{today_date}_attendance.csv")
                    # Initialize the subject attendance file if it doesn't exist
                    if not os.path.exists(subject_attendance_file):
                        with open(subject_attendance_file, 'w', newline='') as subject_file:
                            writer = csv.writer(subject_file)
                            writer.writerow(['Name', 'Time', 'Confidence', 'Method'])
                    # Check if attendance already marked for this student in this subject today
                    already_marked = False
                    with open(subject_attendance_file, 'r') as subject_file:
                        reader = csv.DictReader(subject_file)
                        for row in reader:
                            if row['Name'] == student_name:
                                already_marked = True
                                break
                    if not already_marked and student_name != "Unknown":
                        with open(subject_attendance_file, 'a', newline='') as subject_file:
                            writer = csv.writer(subject_file)
                            writer.writerow([student_name, datetime.now().strftime("%H:%M:%S"), f"{confidence:.2f}%", method])
                        print(f"Marked attendance for {student_name} in {subject} at {current_time} (Confidence: {confidence:.2f}%, Method: {method})")
                    """else:
                        if already_marked:
                            print(f"Attendance already marked for {student_name} in {subject} today.")"""
                    return  # Exit after marking attendance for the current subject
        print(f"No active subject found for the current time: {current_time}")
    
    def get_student_name(self, student_id):
        """Get student name from ID."""
        return self.student_ids.get(student_id, "Unknown")
    
    def calculate_box_overlap(self, box1, box2):
        """Calculate intersection over union of two bounding boxes."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Determine coordinates of intersection rectangle
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        # Calculate area of intersection rectangle
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate area of both boxes
        box1_area = w1 * h1
        box2_area = w2 * h2
        
        # Calculate IoU
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        return iou
    
    def ensemble_prediction(self, face_img, resized_face=None):
        """Combine predictions from multiple recognizers."""
        results = []
        
        # Get prediction from LBPH recognizer
        try:
            lbph_id, lbph_conf = self.lbph_recognizer.predict(face_img)
            lbph_conf = 100 - min(100, lbph_conf)
            results.append((lbph_id, lbph_conf, "LBPH"))
        except Exception as e:
            print(f"LBPH prediction error: {e}")
        
        # Try Eigenfaces and Fisherfaces if we have a resized face
        if resized_face is not None:
            try:
                eigen_id, eigen_conf = self.eigen_recognizer.predict(resized_face)
                eigen_conf = 100 - min(100, eigen_conf / 100)
                results.append((eigen_id, eigen_conf, "Eigenfaces"))
            except Exception as e:
                print(f"Eigenfaces prediction error: {e}")
            
            try:
                fisher_id, fisher_conf = self.fisher_recognizer.predict(resized_face)
                fisher_conf = 100 - min(100, fisher_conf / 100)
                results.append((fisher_id, fisher_conf, "Fisherfaces"))
            except Exception as e:
                print(f"Fisherfaces prediction error: {e}")
        
        if not results:
            return None, 0, "None"
        
        # Weighted voting
        votes = defaultdict(float)
        methods = defaultdict(list)
        
        for pred_id, conf, method in results:
            votes[pred_id] += conf
            methods[pred_id].append(method)
        
        if not votes:
            return None, 0, "None"
        
        # Get ID with highest vote
        predicted_id = max(votes, key=votes.get)
        confidence = votes[predicted_id] / len(results)
        method = "+".join(methods[predicted_id])
        
        return predicted_id, confidence, method
    
    def recognize_with_deep_learning(self, frame, face_location):
        """Recognize face using deep learning model."""
        x, y, w, h = face_location
        
        # Convert to format expected by face_recognition
        top, right, bottom, left = y, x+w, y+h, x
        
        try:
            # Get face encoding
            face_encoding = face_recognition.face_encodings(frame, [(top, right, bottom, left)])
            
            if not face_encoding:
                return "Unknown", 0, "DL-Failed"
                
            face_encoding = face_encoding[0]
            
            # Compare with known faces
            if not self.known_face_encodings:
                return "Unknown", 0, "DL-NoData"
                
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            # Convert distance to confidence (0 distance = 100% confidence)
            confidence = (1 - face_distances[best_match_index]) * 100
            
            if confidence > 60:  # Threshold
                return self.known_face_names[best_match_index], confidence, "DeepLearning"
            else:
                return "Unknown", confidence, "DeepLearning"
                
        except Exception as e:
            print(f"Deep learning recognition error: {e}")
            return "Unknown", 0, "DL-Error"
    
    def detect_spoofing(self, frame, face_region):
        """Basic spoof detection to prevent photo attacks."""
        x, y, w, h = face_region
        face = frame[y:y+h, x:x+w]
        
        # Skip if the face region is invalid
        if face.size == 0:
            return False
        
        try:
            # Convert to different color spaces
            ycrcb = cv2.cvtColor(face, cv2.COLOR_BGR2YCrCb)
            hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
            
            # Extract color channels
            y_channel, cr, cb = cv2.split(ycrcb)
            h, s, v = cv2.split(hsv)
            
            # Calculate texture using LBP
            lbp = self.get_lbp_features(y_channel)
            
            # Calculate frequency domain features
            f_transform = np.fft.fft2(y_channel)
            f_shift = np.fft.fftshift(f_transform)
            magnitude = 20 * np.log(np.abs(f_shift) + 1)  # Add 1 to avoid log(0)
            
            # Check for reflection highlights (real faces have specular highlights)
            highlight_pixels = np.sum(v > 200)
            highlight_ratio = highlight_pixels / (w * h)
            
            # Color variance in chrominance channels (should be higher in real faces)
            cr_std = np.std(cr)
            cb_std = np.std(cb)
            
            # Use simple rule-based classification
            if highlight_ratio < 0.01 and cr_std < 5 and cb_std < 5:
                return True  # Likely a spoof
                
            return False  # Likely real
        except Exception as e:
            print(f"Error in spoof detection: {e}")
            return False  # Default to assuming it's real if detection fails
    
    def get_lbp_features(self, gray_img):
        """Extract Local Binary Pattern features."""
        lbp = np.zeros_like(gray_img)
        
        # Get image dimensions
        h, w = gray_img.shape
        
        # Iterate through the image (excluding borders)
        for i in range(1, h-1):
            for j in range(1, w-1):
                # Get 3x3 neighborhood
                center = gray_img[i, j]
                neighborhood = [gray_img[i-1, j-1], gray_img[i-1, j], gray_img[i-1, j+1],
                                gray_img[i, j+1], gray_img[i+1, j+1], gray_img[i+1, j],
                                gray_img[i+1, j-1], gray_img[i, j-1]]
                
                # Calculate LBP code
                code = 0
                for k, neighbor in enumerate(neighborhood):
                    if neighbor >= center:
                        code += 2**k
                
                lbp[i, j] = code
        
        return lbp
    
    def detect_motion(self, frame):
        """Detect moving regions in the frame."""
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Apply morphological operations to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours of moving regions
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_regions = []
        for contour in contours:
            # Filter small contours
            if cv2.contourArea(contour) < 500:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            motion_regions.append((x, y, w, h))
        
        return motion_regions, fg_mask
    
    def track_faces(self, frame, faces):
        """Track faces between frames to maintain identity."""
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Update existing trackers
        tracker_ids_to_remove = []
        for tracker_id, tracker_data in self.tracked_faces.items():
            # Get the tracker
            tracker = self.trackers[tracker_id]
            
            # Update tracker
            success, box = tracker.update(frame)
            
            if not success:
                # Tracking failed
                tracker_ids_to_remove.append(tracker_id)
                continue
            
            # Convert box to x,y,w,h format and ensure integer coordinates
            x, y, w, h = [int(v) for v in box]
            
            # Update tracked face position
            self.tracked_faces[tracker_id]['box'] = (x, y, w, h)
            self.tracked_faces[tracker_id]['frame_count'] += 1

            # --- NEW: Periodically re-run recognition on tracked face ---
            # Extract the current face region
            gray_face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y:y+h, x:x+w]
            if gray_face.shape[0] > 0 and gray_face.shape[1] > 0:
                aligned_face = self.align_face(gray_face, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), x, y, w, h)
                normalized_face = self.normalize_lighting(aligned_face)
                resized_face = cv2.resize(normalized_face, (100, 100))
                student_id, confidence, method = self.ensemble_prediction(normalized_face, resized_face)
                student_name = self.get_student_name(student_id)
                if self.deep_learning_available:
                    dl_name, dl_confidence, dl_method = self.recognize_with_deep_learning(frame, (x, y, w, h))
                    if dl_confidence > confidence and dl_confidence > self.recognition_threshold:
                        student_name = dl_name
                        confidence = dl_confidence
                        method = dl_method
                display_name = student_name if confidence >= 40 else "Unknown"
                display_method = method if confidence >= 40 else method + "+LowConf"
                # Only update if confidence is higher or crosses threshold
                prev_conf = self.tracked_faces[tracker_id]['confidence']
                prev_name = self.tracked_faces[tracker_id]['name']
                if (confidence > prev_conf) or (prev_conf < 40 and confidence >= 40) or (prev_name == "Unknown" and display_name != "Unknown"):
                    self.tracked_faces[tracker_id]['name'] = display_name
                    self.tracked_faces[tracker_id]['confidence'] = confidence
                    self.tracked_faces[tracker_id]['method'] = display_method
            # --- END NEW ---

            # Draw tracking box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Get tracked student info
            student_name = self.tracked_faces[tracker_id]['name']
            confidence = self.tracked_faces[tracker_id]['confidence']
            method = self.tracked_faces[tracker_id]['method']
            
            # Check if we should mark attendance for this tracked face
            if student_name != "Unknown" and confidence > self.recognition_threshold:
                frames_tracked = self.tracked_faces[tracker_id]['frame_count']
                if frames_tracked >= self.frames_to_average:
                    self.mark_attendance(student_name, confidence, method)
            
            # Display tracked info
            display_text = f"{student_name} ({confidence:.1f}%)"
            cv2.putText(frame, display_text, (x, y-10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Remove failed trackers
        for tracker_id in tracker_ids_to_remove:
            del self.trackers[tracker_id]
            del self.tracked_faces[tracker_id]
        
        # Process new detected faces
        for face_rect in faces:
            x, y, w, h = face_rect
            
            # Check if this face overlaps with any existing tracker
            is_new_face = True
            for tracker_id, tracker_data in self.tracked_faces.items():
                tx, ty, tw, th = tracker_data['box']
                
                # Calculate overlap
                overlap = self.calculate_box_overlap((x,y,w,h), (tx,ty,tw,th))
                if overlap > 0.3:  # If more than 30% overlap
                    is_new_face = False
                    break
            
            if is_new_face:
                # Extract face region
                face_roi = gray_frame[y:y+h, x:x+w]
                
                # Check face quality
                quality_ok, quality_msg = self.assess_face_quality(face_roi)
                if not quality_ok:
                    continue
                
                # Apply face alignment and preprocessing
                aligned_face = self.align_face(face_roi, gray_frame, x, y, w, h)
                normalized_face = self.normalize_lighting(aligned_face)
                
                # Check for spoofing
                if self.detect_spoofing(frame, (x, y, w, h)):
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(frame, "FAKE", (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    continue
                
                # Create a resized face for Eigenfaces/Fisherfaces
                resized_face = cv2.resize(normalized_face, (100, 100))
                
                # Use ensemble of traditional recognizers
                student_id, confidence, method = self.ensemble_prediction(normalized_face, resized_face)
                student_name = self.get_student_name(student_id)
                # If available, also try deep learning recognition
                if self.deep_learning_available:
                    dl_name, dl_confidence, dl_method = self.recognize_with_deep_learning(frame, (x, y, w, h))
                    # If deep learning got a good match, use it instead
                    if dl_confidence > confidence and dl_confidence > self.recognition_threshold:
                        student_name = dl_name
                        confidence = dl_confidence
                        method = dl_method
                # Only set label to Unknown if confidence < 40, but always calculate match
                display_name = student_name if confidence >= 40 else "Unknown"
                display_method = method if confidence >= 40 else method + "+LowConf"
                
                # Create a new tracker (using KCF tracker)
                # Use a compatible tracker based on OpenCV version
                try:
                    tracker = cv2.TrackerKCF_create()
                except:
                    try:
                        # For newer OpenCV versions
                        tracker = cv2.legacy.TrackerKCF_create()
                    except:
                        # Fallback to generic tracker
                        tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, (x, y, w, h))
                
                # Create unique tracker ID
                tracker_id = f"tracker_{len(self.trackers)}"
                
                # Store tracker and associated data
                self.trackers[tracker_id] = tracker
                self.tracked_faces[tracker_id] = {
                    'box': (x, y, w, h),
                    'name': display_name,
                    'confidence': confidence,
                    'method': display_method,
                    'frame_count': 0
                }
                
                # Draw rectangle for new detection
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Display recognition results
                display_text = f"{display_name} ({confidence:.1f}%)"
                cv2.putText(frame, display_text, (x, y-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
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
        
        frame_count = 0
        last_email_time = 0
        
        while True:
            # Capture frame from webcam
            ret, frame = video_capture.read()
            if not ret:
                print("Error: Can't receive frame")
                break
            
            frame_count += 1
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            # Detect motion (optional)
            motion_regions, fg_mask = self.detect_motion(frame)
            
            # Track detected faces
            self.track_faces(frame, faces)
            
            # Display motion mask in corner (debug)
            if fg_mask is not None:
                small_mask = cv2.resize(fg_mask, (160, 120))
                frame[10:130, 10:170] = cv2.cvtColor(small_mask, cv2.COLOR_GRAY2BGR)
            
            # Display the frame
            cv2.imshow('Attendance System', frame)
            
            # TEST: Send attendance email every minute
            """current_time = time.time()
            if current_time - last_email_time > 60:
                print("[TEST] Sending attendance email (test mode, every minute)...")
                self.check_and_send_attendance()
                last_email_time = current_time"""
            
            # Check for end of lecture and send attendance
            self.check_and_send_attendance()
            
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
                
                # Process each face using advanced recognition
                for (x, y, w, h) in faces:
                    # Draw rectangle around the face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Extract face region
                    face_roi = gray[y:y+h, x:x+w]
                    
                    # Check face quality
                    quality_ok, quality_msg = self.assess_face_quality(face_roi)
                    if not quality_ok:
                        cv2.putText(frame, f"Low quality: {quality_msg}", (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        continue
                    
                    # Apply preprocessing
                    aligned_face = self.align_face(face_roi, gray, x, y, w, h)
                    normalized_face = self.normalize_lighting(aligned_face)
                    
                    # Create a resized face for Eigenfaces/Fisherfaces
                    resized_face = cv2.resize(normalized_face, (100, 100))
                    
                    # Use ensemble prediction
                    student_id, confidence, method = self.ensemble_prediction(normalized_face, resized_face)
                    student_name = self.get_student_name(student_id)
                    if self.deep_learning_available:
                        dl_name, dl_confidence, dl_method = self.recognize_with_deep_learning(frame, (x, y, w, h))
                        if dl_confidence > confidence and dl_confidence > self.recognition_threshold:
                            student_name = dl_name
                            confidence = dl_confidence
                            method = dl_method
                    display_name = student_name if confidence >= 40 else "Unknown"
                    display_method = method if confidence >= 40 else method + "+LowConf"
                    if confidence > self.recognition_threshold:
                        self.mark_attendance(display_name, confidence, display_method)
                        display_text = f"{display_name} ({confidence:.1f}%, {display_method})"
                    else:
                        display_text = f"Unknown ({confidence:.1f}%)"
                    
                    # Display name and confidence
                    cv2.putText(frame, display_text, (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Display the image with results
                cv2.imshow(f'Image: {filename}', frame)
                cv2.waitKey(0)  # Wait for key press
        
        cv2.destroyAllWindows()
        print("\nOffline processing complete!")
    
    def check_and_send_attendance(self):
        """Check if any lecture just ended and send attendance as Excel to the subject's email."""
        timetable_file = "timetable.csv"
        if not os.path.exists(timetable_file):
            return
        current_time = datetime.now().strftime("%H:%M")
        today_date = datetime.now().strftime("%Y-%m-%d")
        with open(timetable_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                end_time = row['end_time']
                subject = row['subject']
                email = row.get('email', None)
                """if not email:
                    continue
                
                csv_path = os.path.join(self.attendance_folder, f"{subject}_{today_date}_attendance.csv")
                excel_path = csv_path.replace('.csv', '.xlsx')
                if os.path.exists(csv_path):
                    csv_to_excel(csv_path, excel_path)
                    send_email_with_attachment(email, f"Attendance for {subject} on {today_date}",
                                              f"Please find attached the attendance sheet for {subject} on {today_date}.",
                                              excel_path)
                # If the current time matches the end_time (within 1 minute)
                """
                if current_time == end_time:
                    csv_path = os.path.join(self.attendance_folder, f"{subject}_{today_date}_attendance.csv")
                    excel_path = csv_path.replace('.csv', '.xlsx')
                    if os.path.exists(csv_path):
                        self.csv_to_excel(csv_path, excel_path)
                        self.send_email_with_attachment(email, f"Attendance for {subject} on {today_date}",
                                                      f"Please find attached the attendance sheet for {subject} on {today_date}.",
                                                      excel_path)


def setup_instructions():
    """Print setup instructions for first-time usage."""
    print("\n---- SETUP INSTRUCTIONS ----")
    print("1. Before running this system, install required packages:")
    print("   pip install opencv-contrib-python numpy")
    print("\n2. Add student images to the 'data/students/' folder:")
    print("   - Each image should contain only one clear face")
    print("   - Name the file with the student's name and number (e.g., sainand1.jpg, sainand2.jpg)")
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
        attendance_system = EnhancedAttendanceSystem()
        
        if webcam_works:
            attendance_system.run()
        else:
            print("\nRunning in offline mode since webcam is not available.")
            attendance_system.run_offline_mode()