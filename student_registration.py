import os
import cv2
import tkinter as tk
from tkinter import messagebox
from tkinter import simpledialog
from PIL import Image, ImageTk

# Ensure required directories exist
STUDENTS_FOLDER = "data/students"
TEST_IMAGES_FOLDER = "data/test_images"
os.makedirs(STUDENTS_FOLDER, exist_ok=True)
os.makedirs(TEST_IMAGES_FOLDER, exist_ok=True)

def capture_images(first_name, last_name):
    """Capture 20-25 images of the student and save them."""
    student_name = f"{first_name}_{last_name}"
    student_id = len(os.listdir(STUDENTS_FOLDER)) + 1  # Assign a new ID based on the number of students
    print(f"Assigned ID: {student_id} for {student_name}")

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not access the webcam.")
        return

    count = 0
    while count < 25:
        print(f"Capturing image {count + 1}...")
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # Save the image
        file_name = f"{student_name}{count + 1}.jpg"
        cv2.imwrite(os.path.join(STUDENTS_FOLDER, file_name), frame)
        cv2.imwrite(os.path.join(TEST_IMAGES_FOLDER, file_name), frame)
        count += 1
        # Display the frame
        cv2.imshow("Capturing Images - Press 'q' to Quit", frame)
        show_frame_in_tkinter(frame)
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if count == 25:
        messagebox.showinfo("Success", f"Successfully captured 25 images for {student_name}.")
    else:
        messagebox.showwarning("Incomplete", f"Captured only {count} images for {student_name}.")

def detect_motion(bg_subtractor, frame):
    """Detect moving regions in the frame."""
    fg_mask = bg_subtractor.apply(frame)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion_regions = []
    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        motion_regions.append((x, y, w, h))
    return motion_regions, fg_mask

def detect_faces(gray_frame):
    """Detect faces in the frame using Haar cascades."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    return faces

def show_frame_in_tkinter(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)

def register_student():
    """Register a new student by capturing their images."""
    first_name = simpledialog.askstring("Input", "Enter the student's first name:")
    if not first_name:
        messagebox.showerror("Error", "First name is required.")
        return
    last_name = simpledialog.askstring("Input", "Enter the student's last name:")
    if not last_name:
        messagebox.showerror("Error", "Last name is required.")
        return
    capture_images(first_name.strip(), last_name.strip())

def test_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
    else:
        print("Webcam is working!")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Webcam Test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

# Create the GUI
root = tk.Tk()
root.title("Student Registration")
root.geometry("500x300")
root.configure(bg="#f0f0f5")  # Light gray background

# Style options
LABEL_STYLE = {"font": ("Helvetica", 18, "bold"), "bg": "#f0f0f5", "fg": "#333"}
BUTTON_STYLE = {
    "font": ("Helvetica", 14),
    "bg": "#4CAF50",
    "fg": "white",
    "activebackground": "#45a049",
    "height": 2,
    "width": 40,
    "bd": 0,
    "relief": tk.FLAT,
    "cursor": "hand2",
    "highlightthickness": 0
}

label = tk.Label(root, text="Register New Student", **LABEL_STYLE)
label.pack(pady=20)

register_button = tk.Button(root, text="Register Student", command=register_student, **BUTTON_STYLE)
register_button.pack(pady=10)

exit_button = tk.Button(
    root,
    text="Exit",
    command=root.quit,
    **{**BUTTON_STYLE, "bg": "#f44336", "activebackground": "#d32f2f"}  # Red for exit
)
exit_button.pack(pady=5)

root.mainloop()
