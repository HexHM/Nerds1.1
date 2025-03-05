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

# Initialize lock status for grid
grid_locked = False
locked_grid_area = None

def create_machine_grid(image_shape):
    """Define machine area based on frame dimensions."""
    height, width = image_shape[0], image_shape[1]
    
    # Define machine area (adjust these coordinates for your setup)
    machine_top_left = (int(width * 0.25), int(height * 0.25))
    machine_bottom_right = (int(width * 0.75), int(height * 0.75))
    
    return machine_top_left, machine_bottom_right

def detect_black_grid_lines(frame):
    """
    Detects black grid lines from the ground (if painted) using color filtering and edge detection.
    Returns the detected lines for further processing.
    """
    # Step 1: Convert to HSV to easily isolate black color
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Step 2: Define the HSV range for black color
    lower_black = np.array([0, 0, 0])  # Lower bound of black in HSV space
    upper_black = np.array([180, 255, 50])  # Upper bound of black in HSV space (low value)

    # Step 3: Create a mask for the black regions
    mask = cv2.inRange(hsv, lower_black, upper_black)

    # Step 4: Use Canny edge detection to highlight edges (grid lines)
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)

    # Step 5: Use Hough Line Transform to detect straight lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    
    return lines

def draw_grid_from_lines(frame, lines):
    """
    Draws detected grid lines onto the frame.
    """
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green grid lines
    return frame

def detect_workers(frame, worker_hsv_color):
    """
    Detect workers based on the selected color.
    """
    # Calculate the color range (tolerance) around the selected color
    lower_bound = np.array([worker_hsv_color[0] - 10, 100, 100])  # lower range of color
    upper_bound = np.array([worker_hsv_color[0] + 10, 255, 255])  # upper range of color
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create mask for worker color
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out small contours (noise)
    worker_contours = [contour for contour in contours if cv2.contourArea(contour) > 100]
    
    return worker_contours

def is_worker_too_close(worker_contours, machine_area):
    """Check if any worker is too close to the machine area."""
    machine_top_left, machine_bottom_right = machine_area
    machine_center = (
        (machine_top_left[0] + machine_bottom_right[0]) // 2,
        (machine_top_left[1] + machine_bottom_right[1]) // 2
    )
    
    for contour in worker_contours:
        # Get center of worker contour
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
            
        worker_x = int(M["m10"] / M["m00"])
        worker_y = int(M["m01"] / M["m00"])
        
        # Calculate distance from worker to machine center
        distance = np.sqrt((worker_x - machine_center[0])**2 + (worker_y - machine_center[1])**2)
        
        # Debug information
        print(f"Worker detected at ({worker_x}, {worker_y}), distance: {distance:.2f} pixels")
        
        if distance < SAFETY_DISTANCE_THRESHOLD:
            return True
    
    return False

def draw_results(frame, worker_contours, lines, machine_area, machine_enabled):
    # Draw grid lines detected from the ground if grid is not locked
    if not grid_locked:
        frame = draw_grid_from_lines(frame, lines)
    else:
        # If grid is locked, draw the locked grid area
        machine_top_left, machine_bottom_right = machine_area
        color = (0, 255, 0) if machine_enabled else (0, 0, 255)
        cv2.rectangle(frame, machine_top_left, machine_bottom_right, color, 2)

    # Draw machine area
    machine_top_left, machine_bottom_right = machine_area
    color = (0, 255, 0) if machine_enabled else (0, 0, 255)
    cv2.rectangle(frame, machine_top_left, machine_bottom_right, color, 2)
    
    # Draw safety zone
    machine_center = (
        (machine_top_left[0] + machine_bottom_right[0]) // 2,
        (machine_top_left[1] + machine_bottom_right[1]) // 2
    )
    cv2.circle(frame, machine_center, SAFETY_DISTANCE_THRESHOLD, (0, 255, 255), 2)
    
    # Draw detected workers
    for contour in worker_contours:
        cv2.drawContours(frame, [contour], -1, (0, 0, 255), 2)
        
        # Calculate and display center of contour
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cx, cy), 5, (255, 0, 255), -1)
    
    # Add status text
    status = "SAFE: Machine ENABLED" if machine_enabled else "ALERT: Worker TOO CLOSE - Machine DISABLED"
    color = (0, 255, 0) if machine_enabled else (0, 0, 255)
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return frame

def control_machine(enable):
    if enable:
        GPIO.output(MACHINE_RELAY_PIN, GPIO.HIGH)
        print("Machine ENABLED")
    else:
        GPIO.output(MACHINE_RELAY_PIN, GPIO.LOW)
        print("Machine DISABLED - Worker too close!")

try:
    last_machine_state = True
    # Select a color (e.g., red) to detect workers in HSV color space
    selected_color = (0, 100, 100)  # Example: Red color in HSV space

    while True:
        # Capture frame
        frame = picam2.capture_array()

        # Detect the black grid lines from the frame (if grid is not locked)
        lines = detect_black_grid_lines(frame) if not grid_locked else None

        # Define machine area based on frame dimensions
        machine_area = create_machine_grid(frame.shape)

        # Detect workers based on the dynamically selected color
        worker_contours = detect_workers(frame, selected_color)

        # Check if any worker is too close
        worker_too_close = is_worker_too_close(worker_contours, machine_area)

        # Control the machine
        machine_enabled = not worker_too_close
        if machine_enabled != last_machine_state:
            control_machine(machine_enabled)
            last_machine_state = machine_enabled

        # Draw results on frame (including grid, machine area, and worker detection)
        output_frame = draw_results(frame, worker_contours, lines, machine_area, machine_enabled)

        # Display the resulting frame
        cv2.imshow('Worker Safety Detection', output_frame)

        # Save frame with timestamp if a worker is too close (for logging purposes)
        if worker_too_close:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            cv2.imwrite(f"safety_alert_{timestamp}.jpg", output_frame)

        # Press 'l' to lock the grid area
        if cv2.waitKey(1) & 0xFF == ord('l') and not grid_locked:
            grid_locked = True
            locked_grid_area = machine_area
            print("Grid locked!")

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Maintain frame rate
        time.sleep(1 / FPS)

except KeyboardInterrupt:
    print("Program terminated by user")
finally:
    # Clean up
    GPIO.cleanup()
    cv2.destroyAllWindows()
    picam2.stop()
    print("System shutdown safely")
