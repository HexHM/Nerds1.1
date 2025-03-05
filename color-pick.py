import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# Global variable to store the RGB color
picked_color = None
cap = None

def pick_color(event, x, y, flags, param):
    global picked_color, frame

    if event == cv2.EVENT_LBUTTONDOWN:
        # Capture the RGB color at the clicked position
        picked_color = frame[y, x]
        messagebox.showinfo("Color Picked", f"RGB: {picked_color}")

def display_frame():
    global frame

    # Read the next frame from the video
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        return

    # Convert the frame to RGB (OpenCV uses BGR by default)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the frame to Image object for Tkinter display
    img = Image.fromarray(frame_rgb)
    img_tk = ImageTk.PhotoImage(img)

    # Display the frame in Tkinter canvas
    canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
    canvas.image = img_tk  # Keep a reference to the image to avoid garbage collection

    # Update Tkinter window
    window.after(30, display_frame)  # Delay for 30ms and call display_frame again

def on_pick_color_button_click():
    global cap
    # Start capturing from the webcam (live video feed)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Failed to open webcam")
        return
    # Start displaying video frames when the button is pressed
    display_frame()

# Create the main window using Tkinter
window = tk.Tk()
window.title("Pick RGB Color from Live Video Feed")

# Create a Tkinter canvas to display the video
canvas = tk.Canvas(window, width=640, height=480)
canvas.pack()

# Add a button to pick the color
pick_color_button = tk.Button(window, text="Pick Color", command=on_pick_color_button_click)
pick_color_button.pack()

# Set up OpenCV mouse callback function for color picking
cv2.namedWindow("Video Feed")
cv2.setMouseCallback("Video Feed", pick_color)

# Start the Tkinter event loop
window.mainloop()

# Release the video capture once the window is closed
if cap:
    cap.release()
cv2.destroyAllWindows()