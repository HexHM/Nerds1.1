import cv2
import numpy as np
import time
import RPi.GPIO as GPIO
from picamera2 import Picamera2

# Configuration
SAFETY_DISTANCE_THRESHOLD = 200  # Adjust based on your setup (in pixels)
MACHINE_RELAY_PIN = 17  # GPIO pin connected to machine control relay
CAMERA_RESOLUTION = (640, 480)
FPS = 10

# Initialize GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(MACHINE_RELAY_PIN, GPIO.OUT)
GPIO.output(MACHINE_RELAY_PIN, GPIO.HIGH)  # Machine enabled by default

# Initialize camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": CAMERA_RESOLUTION})
picam2.configure(config)
picam2.start()

# Allow camera to warm up
time.sleep(2)

# List to store clicked points
grid_locked = False
locked_grid = None
points = []

# Capture a frame from the camera
frame = picam2.capture_array()

def select_points(event, x, y, flags, param):
    """Mouse callback function to store four corner points."""
    global points, grid_locked, locked_grid
    
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        print(f"Point {len(points)} selected: {x}, {y}")

    if len(points) == 4 and not grid_locked:
        print("Grid locked!")
        grid_locked = True
        locked_grid = points.copy()
        draw_grid()


def draw_grid():
    global locked_grid, frame
    if locked_grid is None:
        return

    grid_size = (5, 5)  # 5x5 grid
    src_pts = np.array(locked_grid, dtype=np.float32)
    dst_pts = np.array([
        [0, 0], [grid_size[0] - 1, 0],
        [0, grid_size[1] - 1], [grid_size[0] - 1, grid_size[1] - 1]
    ], dtype=np.float32) * 100  # Scale for visualization
    
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(frame, M, (grid_size[0] * 100, grid_size[1] * 100))
    
    for i in range(1, grid_size[0]):
        cv2.line(warped, (i * 100, 0), (i * 100, grid_size[1] * 100), (0, 255, 0), 2)
    for j in range(1, grid_size[1]):
        cv2.line(warped, (0, j * 100), (grid_size[0] * 100, j * 100), (0, 255, 0), 2)
    
    cv2.imshow('Warped Grid', warped)

def detect_workers(frame, worker_hsv_color):
    lower_bound = np.array([worker_hsv_color[0] - 10, 100, 100])
    upper_bound = np.array([worker_hsv_color[0] + 10, 255, 255])
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [contour for contour in contours if cv2.contourArea(contour) > 100]

def is_worker_too_close(worker_contours):
    for contour in worker_contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            distance = np.sqrt(cx**2 + cy**2)
            if distance < SAFETY_DISTANCE_THRESHOLD:
                return True
    return False

def draw_detected_worker(frame, worker_contours):
    for contour in worker_contours:
        cv2.drawContours(frame, [contour], -1, (0, 0, 255), 2)

def break_function():
    print("Breaking program...")
    GPIO.cleanup()
    picam2.stop()
    cv2.destroyAllWindows()
    exit()

cv2.imshow('Select Grid Corners', frame)
cv2.setMouseCallback('Select Grid Corners', select_points)

print("Click on four corners of the grid area in order (top-left, top-right, bottom-left, bottom-right)")
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(points) != 4:
    print("Error: You must select exactly 4 points!")
    exit()

draw_grid()
cv2.waitKey(0)
cv2.destroyAllWindows()

def control_machine(enable):
    if enable:
        GPIO.output(MACHINE_RELAY_PIN, GPIO.HIGH)
        print("Machine ENABLED")
    else:
        GPIO.output(MACHINE_RELAY_PIN, GPIO.LOW)
        print("Machine DISABLED - Worker too close!")

# Cleanup
break_function()
