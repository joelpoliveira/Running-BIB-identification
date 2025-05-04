import os
import cv2
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

def identify_bibs(image_path, weights_path):
    """
    Locate and identify bibs in a photo and display the photo with bounding boxes.

    Args:
        image_path (str): Path to the input image.
        weights_path (str): Path to the YOLO weights file.

    Returns:
        None
    """
    # Load YOLO model
    model = YOLO(weights_path)

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    # Convert BGR image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform inference
    results = model(rgb_image)

    # Extract bounding boxes and labels
    detections = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, confidence, class]

    for detection in detections:
        x1, y1, x2, y2, confidence, cls = detection
        label = f"Bib {int(cls)}: {confidence:.2f}"

        # Draw bounding box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Put label text
        cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the image with bounding boxes
    cv2.imshow('BIB Identification', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    image_path = "path_to_your_image.jpg"  # Replace with your image path
    weights_path = os.getenv("MODEL_PATH")  # Replace with your YOLO weights file

    if weights_path is None:
        raise ("Error: YOLO weights path not set in environment variables.")
    # Call the function to identify bibs
    identify_bibs(image_path, weights_path)