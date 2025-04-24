# README.md
# Auto Attendance System

A simple face recognition-based attendance system that identifies students from a webcam feed and marks their attendance.

## Setup

1. Install required dependencies:
   ```
   pip install face_recognition opencv-python numpy
   ```

2. Add student images to the `data/students/` folder:
   - Each image should contain only one face
   - Name the image file with the student's name (e.g., john_smith.jpg)

## Running the System

1. Run the main script:
   ```
   python main.py
   ```

2. The system will:
   - Load student images from the data folder
   - Start the webcam
   - Identify faces in the frame
   - Mark attendance for recognized students
   - Display the results in real-time

3. Press 'q' to quit the program.

## Attendance Records

Attendance records are saved in the `attendance_records/` folder as CSV files, with one file per day.