import cv2
import numpy as np
import time
import RPi.GPIO as GPIO
from picamera2 import Picamera2

# Configuration
SAFETY_DISTANCE_THRESHOLD = 200
MACHINE_RELAY_PIN = 17
CAMERA_RESOLUTION = (640, 480)
FPS = 10

# Initialize GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(MACHINE_RELAY_PIN, GPIO.OUT)
GPIO.output(MACHINE_RELAY_PIN, GPIO.HIGH)

# Initialize camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": CAMERA_RESOLUTION})
picam2.configure(config)
picam2.start()

time.sleep(2)

# Global variables for grid
grid_points = []
grid_size = 50  # Default grid size
grid_locked = False

# Mouse callback function to select grid corners
def select_grid(event, x, y, flags, param):
    global grid_points, grid_locked
    if event == cv2.EVENT_LBUTTONDOWN and not grid_locked:
        if len(grid_points) < 4:
            grid_points.append((x, y))
            print(f"Grid point selected: {x}, {y}")
        if len(grid_points) == 4:
            print("Grid fully defined. Press 'l' to lock it.")

def draw_grid(frame):
    if len(grid_points) == 4:
        top_left, top_right, bottom_right, bottom_left = grid_points
        
        rows = (bottom_left[1] - top_left[1]) // grid_size
        cols = (top_right[0] - top_left[0]) // grid_size
        
        for i in range(rows + 1):
            y = top_left[1] + i * grid_size
            cv2.line(frame, (top_left[0], y), (top_right[0], y), (0, 255, 0), 2)
        
        for j in range(cols + 1):
            x = top_left[0] + j * grid_size
            cv2.line(frame, (x, top_left[1]), (x, bottom_left[1]), (0, 255, 0), 2)
    return frame

cv2.namedWindow('Worker Safety Detection')
cv2.setMouseCallback('Worker Safety Detection', select_grid)

# Trackbar callback function
def update_grid_size(val):
    global grid_size
    grid_size = max(10, val)  # Ensure grid size is at least 10 pixels

cv2.createTrackbar('Grid Size', 'Worker Safety Detection', 50, 200, update_grid_size)

try:
    last_machine_state = True
    while True:
        frame = picam2.capture_array()
        frame = cv2.flip(frame, -1)
        
        if grid_points:
            frame = draw_grid(frame)
        
        cv2.imshow('Worker Safety Detection', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('l') and len(grid_points) == 4:
            grid_locked = True
            print("Grid locked!")
        elif key == ord('q'):
            break

        time.sleep(1 / FPS)

except KeyboardInterrupt:
    print("Program terminated by user")
finally:
    GPIO.cleanup()
    cv2.destroyAllWindows()
    picam2.stop()
    print("System shutdown safely")
   