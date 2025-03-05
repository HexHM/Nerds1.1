import cv2
import numpy as np
import time
import RPi.GPIO as GPIO
from picamera2 import Picamera2

# Configuration
SAFETY_DISTANCE_THRESHOLD = 200  # Adjust as per your safety requirements
MACHINE_RELAY_PIN = 17
CAMERA_RESOLUTION = (1280, 720)  # You can change this to your preference
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

# Global variables for grid and color tracking
grid_points = []
grid_size = 50  # Default grid size
worker_color = None
machine_color = None
freeze_frame = None
grid_locked = False

# Mouse callback function to select grid corners
def select_grid(event, x, y, flags, param):
    global grid_points, freeze_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        if freeze_frame is not None:
            # No color selection logic here now, just grid selection
            if len(grid_points) < 2:
                grid_points.append((x, y))
                print(f"Grid point selected: {x}, {y}")
            if len(grid_points) == 2:
                print("Grid fully defined. Press 'l' to lock it.")
            freeze_frame = None
        else:
            if len(grid_points) < 2:
                grid_points.append((x, y))
                print(f"Grid point selected: {x}, {y}")

cv2.namedWindow('Worker Safety Detection')
cv2.setMouseCallback('Worker Safety Detection', select_grid)

# Trackbar callback function
def update_grid_size(val):
    global grid_size
    grid_size = max(10, val)  # Ensure grid size is at least 10 pixels

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

# Function to highlight a 4x4 grid area centered around the worker
def highlight_warning_area(frame, position):
    x, y = position
    warning_area_size = grid_size * 4
    top_left_x = x - warning_area_size // 2
    top_left_y = y - warning_area_size // 2
    bottom_right_x = top_left_x + warning_area_size
    bottom_right_y = top_left_y + warning_area_size

    cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 0, 255), 2)  # Red warning box

# Function to highlight the 4x4 grid area around the worker, constrained within the grid
def highlight_grid_around_worker(frame, worker_position):
    x, y = worker_position
    # Calculate the grid coordinates that will contain the worker (in terms of grid_size)
    grid_x = (x // grid_size) * grid_size
    grid_y = (y // grid_size) * grid_size

    # Define the size of the area to highlight (4x4 grid blocks)
    highlight_size = grid_size * 4

    # Make sure the highlight does not go outside the grid area
    x_end = min(grid_x + highlight_size, grid_points[1][0])
    y_end = min(grid_y + highlight_size, grid_points[1][1])

    # Highlight the surrounding grid area
    for dx in range(0, x_end - grid_x, grid_size):
        for dy in range(0, y_end - grid_y, grid_size):
            top_left_x = grid_x + dx
            top_left_y = grid_y + dy
            bottom_right_x = top_left_x + grid_size
            bottom_right_y = top_left_y + grid_size
            cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 255), 2)  # Yellow for highlighting

# Function to check if a machine is too close to a worker (based on distance)
def check_machine_warning(frame, machine_positions, worker_positions):
    for wx, wy in worker_positions:
        worker_grid_x = (wx // grid_size) * grid_size
        worker_grid_y = (wy // grid_size) * grid_size
        warning_area = [(worker_grid_x + dx, worker_grid_y + dy) for dx in range(-2 * grid_size, 3 * grid_size, grid_size) for dy in range(-2 * grid_size, 3 * grid_size, grid_size)]
        for x, y in warning_area:
            cv2.rectangle(frame, (x, y), (x + grid_size, y + grid_size), (0, 0, 255), 2)
        for mx, my in machine_positions:
            if any(x <= mx < x + grid_size and y <= my < y + grid_size for x, y in warning_area):
                cv2.putText(frame, 'WARNING!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                GPIO.output(MACHINE_RELAY_PIN, GPIO.LOW)  # Trigger machine off
                return
    GPIO.output(MACHINE_RELAY_PIN, GPIO.HIGH)

# Main loop
try:
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
            x_start, x_end = sorted([x1, x2])
            y_start, y_end = sorted([y1, y2])

            for x in range(x_start, x_end - grid_size + 1, grid_size):
                for y in range(y_start, y_end - grid_size + 1, grid_size):
                    cv2.rectangle(display_frame, (x, y), (x + grid_size, y + grid_size), (255, 255, 255), 1)
        
        worker_contours = detect_objects(display_frame, worker_color)
        machine_contours = detect_objects(display_frame, machine_color)
        
        worker_positions = [(int(cv2.moments(contour)["m10"] / cv2.moments(contour)["m00"]), int(cv2.moments(contour)["m01"] / cv2.moments(contour)["m00"])) for contour in worker_contours if cv2.moments(contour)["m00"] != 0]
        machine_positions = [(int(cv2.moments(contour)["m10"] / cv2.moments(contour)["m00"]), int(cv2.moments(contour)["m01"] / cv2.moments(contour)["m00"])) for contour in machine_contours if cv2.moments(contour)["m00"] != 0]

        # Highlight worker and machine positions
        for wx, wy in worker_positions:
            cv2.drawContours(display_frame, [worker_contours[worker_positions.index((wx, wy))]], -1, (0, 255, 0), 2)  # Green for worker
            highlight_grid_around_worker(display_frame, (wx, wy))  # Draw 4x4 grid around worker
        
        for mx, my in machine_positions:
            cv2.drawContours(display_frame, [machine_contours[machine_positions.index((mx, my))]], -1, (255, 0, 0), 2)  # Blue for machine
        
        # Check for warning (worker too close to machine)
        check_machine_warning(display_frame, machine_positions, worker_positions)
        
        cv2.imshow('Worker Safety Detection', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('l') and len(grid_points) == 2:
            grid_locked = True
            print("Grid locked!")
        elif key == ord('q'):
            break
        elif key == ord('c'):  # Select worker color
            freeze_frame = frame.copy()
            print("Frame frozen. Click on a worker to select color.")
        elif key == ord('m'):  # Select machine color
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
