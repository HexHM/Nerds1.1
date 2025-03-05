import cv2
import numpy as np
import time
import RPi.GPIO as GPIO
from picamera2 import Picamera2

# Configuration
MACHINE_RELAY_PIN = 17
CAMERA_RESOLUTION = (1280, 720)
FPS = 60

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

# Global variables
grid_points = []
grid_size = 50  # Default grid cell size
grid_cells = []

# Mouse callback function to select grid corners
def select_grid_or_color(event, x, y, flags, param):
    global grid_points, grid_cells, grid_size
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(grid_points) < 2:
            grid_points.append((x, y))
            print(f"Grid point selected: {x}, {y}")
        if len(grid_points) == 2:
            generate_grid()

# Function to generate the grid inside the selected area
def generate_grid():
    global grid_cells, grid_points, grid_size
    grid_cells = []
    if len(grid_points) == 2:
        x1, y1 = grid_points[0]
        x2, y2 = grid_points[1]
        x_start, x_end = sorted([x1, x2])
        y_start, y_end = sorted([y1, y2])
        for x in range(x_start, x_end - grid_size + 1, grid_size):
            for y in range(y_start, y_end - grid_size + 1, grid_size):
                grid_cells.append((x, y))
        print(f"Grid generated with {len(grid_cells)} cells.")

# Function to update grid size from the slider
def update_grid_size(val):
    global grid_size
    grid_size = max(10, val)  # Ensure grid size is at least 10 pixels
    generate_grid()  # Regenerate the grid when the size is updated

cv2.namedWindow('Worker Safety Detection')
cv2.setMouseCallback('Worker Safety Detection', select_grid_or_color)

# Create a slider for grid size adjustment
cv2.createTrackbar('Grid Size', 'Worker Safety Detection', 50, 200, update_grid_size)

# Function to detect objects based on color
def detect_objects(frame, target_hsv_color):
    if target_hsv_color is None:
        return []
    lower_bound = np.array([target_hsv_color[0] - 10, 100, 100])
    upper_bound = np.array([target_hsv_color[0] + 10, 255, 255])
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [contour for contour in contours if cv2.contourArea(contour) > 100]

# Main loop for object detection and handling
try:
    while True:
        frame = picam2.capture_array()
        frame = cv2.flip(frame, -1)
        
        display_frame = frame.copy()

        # If grid is defined, draw it on the frame
        if len(grid_points) == 2:
            for x, y in grid_cells:
                cv2.rectangle(display_frame, (x, y), (x + grid_size, y + grid_size), (255, 255, 255), 1)

        # Detect worker and machine objects based on color (mockup)
        worker_color = np.array([60, 255, 255])  # Example worker color in HSV (Yellowish)
        machine_color = np.array([120, 255, 255])  # Example machine color in HSV (Bluish)
        
        worker_contours = detect_objects(display_frame, worker_color)
        machine_contours = detect_objects(display_frame, machine_color)
        
        # Draw detected contours for workers (green) and machines (red)
        for contour in worker_contours:
            cv2.drawContours(display_frame, [contour], -1, (0, 255, 0), 2)
        
        for contour in machine_contours:
            cv2.drawContours(display_frame, [contour], -1, (0, 0, 255), 2)

        # Display the result
        cv2.imshow('Worker Safety Detection', display_frame)

        # Handle keypress events
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        time.sleep(1 / FPS)

except KeyboardInterrupt:
    print("Program terminated by user")
finally:
    GPIO.cleanup()
    cv2.destroyAllWindows()
    picam2.stop()
    print("System shutdown safely")
