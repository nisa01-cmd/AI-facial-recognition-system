import cv2
from deepface import DeepFace
import threading
import tkinter as tk
from PIL import Image, ImageTk

# Start the GUI
window = tk.Tk()
window.title("Facial Analysis System")

# Create a label to show the video feed
video_label = tk.Label(window)
video_label.pack()

# Start video capture
cap = cv2.VideoCapture(0)

def analyze_frame():
    ret, frame = cap.read()
    if not ret:
        return

    try:
        results = DeepFace.analyze(
            frame,
            actions=['age', 'gender', 'emotion', 'race'],
            enforce_detection=False
        )

        for face in results:
            print(f"[INFO] {face}")  # <--- shows detailed output in terminal


            if 'region' in face:
                x = face['region']['x']
                y = face['region']['y']
                w = face['region']['w']
                h = face['region']['h']

                age = face['age']
                gender = face['dominant_gender']
                emotion = face['dominant_emotion']
                race = face['dominant_race']

                # Draw box and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"{gender}, Age: {age}, {emotion}, {race}"
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    except Exception as e:
        print("[ERROR]", e)

    # Show in Tkinter
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    window.after(200, analyze_frame)  # Delay increased for clarity


# Start the loop
analyze_frame()
window.mainloop()

# Release the camera when window closes
cap.release()
cv2.destroyAllWindows()
