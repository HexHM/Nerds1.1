import cv2
import numpy as np
import tensorflow as tf

# Load the TensorFlow Lite model
def load_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Preprocess the image for the TensorFlow Lite model
def preprocess_image(frame, input_size=(300, 300)):
    image_resized = cv2.resize(frame, input_size)
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_np = np.expand_dims(image_rgb, axis=0).astype(np.float32)
    return image_np

# Post-process the detections
def post_process(frame, output_data, conf_threshold=0.2):
    height, width, _ = frame.shape

    # Extract detection information (bounding boxes, class IDs, and confidence scores)
    boxes = output_data[0]
    classes = output_data[1]
    scores = output_data[2]

    for i in range(len(scores)):
        if scores[i] > conf_threshold:
            # Extract box coordinates (normalized)
            ymin, xmin, ymax, xmax = boxes[i]

            # Convert to absolute coordinates
            (startX, startY, endX, endY) = (
                int(xmin * width),
                int(ymin * height),
                int(xmax * width),
                int(ymax * height),
            )

            # Draw the bounding box
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            # Add label text (confidence score)
            cv2.putText(
                frame,
                f"Confidence: {scores[i] * 100:.2f}%",
                (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
    
    return frame

# Capture and process video frames
def main():
    # Load the model
    interpreter = load_model('ssd_mobilenet_v2_coco.tflite')  # Path to your downloaded .tflite file

    # Get model input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Initialize camera (0 is the default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Preprocess the image for TensorFlow Lite model
        input_data = preprocess_image(frame)

        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Run inference
        interpreter.invoke()

        # Get output tensor (bounding boxes, classes, and scores)
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding boxes
        classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class IDs
        scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence scores

        # Post-process and draw bounding boxes on the frame
        frame = post_process(frame, [boxes, classes, scores])

        # Display the frame
        cv2.imshow('Object Detection', frame)

        # Exit loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
