import cv2
import numpy as np

# Load the pre-trained MobileNet SSD model (used for human and object detection)
def load_model():
    # Load the Caffe model and configuration files
    net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')
    return net

# Initialize the camera
def initialize_camera():
    # Use the default camera (0), or replace with other camera IDs if needed
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return None
    return cap

# Process each frame and detect objects (humans or machines)
def process_frame(frame, net):
    # Prepare the frame for detection (resize and normalize it)
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5, 127.5, 127.5, False)
    net.setInput(blob)
    
    # Run the forward pass to get the detection results
    detections = net.forward()
    
    # Loop through each detection in the frame
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # Only proceed if the confidence is above a threshold
        if confidence > 0.2:
            # Get the coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Draw the bounding box around the object (human or machine)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            
            # Optionally, put the label on the object (e.g., "Person")
            cv2.putText(frame, f"Confidence: {confidence*100:.2f}%", 
                        (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 2)
    
    return frame

# Display the result frame
def display_frame(frame):
    cv2.imshow("Detection Frame", frame)
    
    # Wait for the user to press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False
    return True

# Clean up resources after the program finishes
def cleanup(cap):
    cap.release()
    cv2.destroyAllWindows()

# Main function to run the detection loop
def main():
    # Load pre-trained model for detection
    net = load_model()

    # Initialize camera
    cap = initialize_camera()
    if cap is None:
        return
    
    while True:
        # Capture each frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        
        # Process the frame for detection
        frame = process_frame(frame, net)
        
        # Display the frame with detection results
        if not display_frame(frame):
            break
    
    # Clean up and release resources
    cleanup(cap)

if __name__ == "__main__":
    main()
