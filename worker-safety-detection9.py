import cv2
import numpy as np
import time
import RPi.GPIO as GPIO
from picamera2 import Picamera2

# Configuration
SAFETY_DISTANCE_THRESHOLD = 200
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

# Global variables for grid
grid_points = []
grid_size = 50  # Default grid size
grid_locked = False
worker_color = None
machine_color = None
freeze_frame = None
grid_cells = []

def generate_grid():
    global grid_cells
    grid_cells = []
    if len(grid_points) == 2:
        x1, y1 = grid_points[0]
        x2, y2 = grid_points[1]
        x_start, x_end = sorted([x1, x2])
        y_start, y_end = sorted([y1, y2])
        for x in range(x_start, x_end, grid_size):
            for y in range(y_start, y_end, grid_size):
                grid_cells.append((x, y))

def select_grid_or_color(event, x, y, flags, param):
    global grid_points, grid_locked, worker_color, machine_color, freeze_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        if freeze_frame is not None:
            hsv_frame = cv2.cvtColor(freeze_frame, cv2.COLOR_BGR2HSV)
            if worker_color is None:
                worker_color = hsv_frame[y, x]
                print(f"Worker color selected: {worker_color}")
            else:
                machine_color = hsv_frame[y, x]
                print(f"Machine color selected: {machine_color}")
            freeze_frame = None
        elif not grid_locked and len(grid_points) < 2:
            grid_points.append((x, y))
            print(f"Grid point selected: {x}, {y}")
        if len(grid_points) == 2:
            generate_grid()
            print("Grid fully defined. Press 'l' to lock it.")

cv2.namedWindow('Worker Safety Detection')
cv2.setMouseCallback('Worker Safety Detection', select_grid_or_color)

def update_grid_size(val):
    global grid_size
    grid_size = max(10, val)
    if len(grid_points) == 2:
        generate_grid()

cv2.createTrackbar('Grid Size', 'Worker Safety Detection', 50, 200, update_grid_size)

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

def highlight_grid_area(frame, positions):
    highlighted_cells = set()
    for cx, cy in positions:
        for x, y in grid_cells:
            if x <= cx < x + grid_size and y <= cy < y + grid_size:
                if (x, y) not in highlighted_cells:
                    cv2.rectangle(frame, (x, y), (x + grid_size, y + grid_size), (0, 0, 255), 2)
                    highlighted_cells.add((x, y))
                break

try:
    last_machine_state = True
    while True:
        frame = picam2.capture_array()
        frame = cv2.flip(frame, -1)
        
        if freeze_frame is not None:
            display_frame = freeze_frame.copy()
        else:
            display_frame = frame.copy()
        
        if len(grid_points) == 2:
            x1, y1 = grid_points[0]
            x2, y2 = grid_points[1]
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for x, y in grid_cells:
                cv2.rectangle(display_frame, (x, y), (x + grid_size, y + grid_size), (255, 255, 255), 1)
        
        worker_contours = detect_objects(display_frame, worker_color)
        machine_contours = detect_objects(display_frame, machine_color)
        
        worker_positions = []
        for contour in worker_contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                worker_positions.append((cx, cy))
        
        highlight_grid_area(display_frame, worker_positions)
        
        for contour in machine_contours:
            cv2.drawContours(display_frame, [contour], -1, (255, 0, 0), 2)
        
        cv2.imshow('Worker Safety Detection', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('l') and len(grid_points) == 2:
            grid_locked = True
            print("Grid locked!")
        elif key == ord('q'):
            break
        elif key == ord('c'):
            freeze_frame = frame.copy()
            print("Frame frozen. Click on a worker to select color.")
        elif key == ord('m'):
            freeze_frame = frame.copy()
            print("Frame frozen. Click on the machine to select color.")
        
        time.sleep(1 / FPS)

except KeyboardInterrupt:
    print("Program terminated by user")
finally:
    GPIO.cleanup()
    cv2.destroyAllWindows()
    picam2.stop()
    print("System shutdown safely")
