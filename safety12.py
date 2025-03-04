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

# Global variables
grid_points = []
grid_size = 50  # Grid cell size
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
        for x in range(x_start, x_end - grid_size + 1, grid_size):
            for y in range(y_start, y_end - grid_size + 1, grid_size):
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

def highlight_grid_area(frame, positions, color):
    highlighted_cells = set()
    for cx, cy in positions:
        for x, y in grid_cells:
            if x <= cx < x + grid_size and y <= cy < y + grid_size:
                if (x, y) not in highlighted_cells:
                    cv2.rectangle(frame, (x, y), (x + grid_size, y + grid_size), color, 2)
                    highlighted_cells.add((x, y))
                break

def check_machine_warning(frame, machine_positions, worker_positions):
    for wx, wy in worker_positions:
        worker_grid_x = (wx // grid_size) * grid_size
        worker_grid_y = (wy // grid_size) * grid_size
        warning_area = [(worker_grid_x + dx, worker_grid_y + dy) for dx in range(-2 * grid_size, 3 * grid_size, grid_size) for dy in range(-2 * grid_size, 3 * grid_size, grid_size)]
        
        for mx, my in machine_positions:
            if any(x <= mx < x + grid_size and y <= my < y + grid_size for x, y in warning_area):
                cv2.putText(frame, 'WARNING!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                GPIO.output(MACHINE_RELAY_PIN, GPIO.LOW)
                return
    GPIO.output(MACHINE_RELAY_PIN, GPIO.HIGH)

try:
    while True:
        frame = picam2.capture_array()
        frame = cv2.flip(frame, -1)
        display_frame = frame.copy()
        
        if len(grid_points) == 2:
            for x, y in grid_cells:
                cv2.rectangle(display_frame, (x, y), (x + grid_size, y + grid_size), (255, 255, 255), 1)
        
        worker_contours = detect_objects(display_frame, worker_color)
        machine_contours = detect_objects(display_frame, machine_color)
        
        worker_positions = [(int(cv2.moments(contour)["m10"] / cv2.moments(contour)["m00"]), int(cv2.moments(contour)["m01"] / cv2.moments(contour)["m00"])) for contour in worker_contours if cv2.moments(contour)["m00"] != 0]
        machine_positions = [(int(cv2.moments(contour)["m10"] / cv2.moments(contour)["m00"]), int(cv2.moments(contour)["m01"] / cv2.moments(contour)["m00"])) for contour in machine_contours if cv2.moments(contour)["m00"] != 0]
        
        highlight_grid_area(display_frame, worker_positions, (0, 255, 0))
        highlight_grid_area(display_frame, machine_positions, (255, 0, 0))
        check_machine_warning(display_frame, machine_positions, worker_positions)
        
        cv2.imshow('Worker Safety Detection', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(1 / FPS)

except KeyboardInterrupt:
    print("Program terminated by user")
finally:
    GPIO.cleanup()
    cv2.destroyAllWindows()
    picam2.stop()
    print("System shutdown safely")
