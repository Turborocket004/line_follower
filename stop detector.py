import cv2
import numpy as np

def detect_stop_symbol(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny Edge Detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
    
    horizontal_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 10:  # Horizontal line check
                horizontal_lines.append((x1, y1, x2, y2))

    # Check if two horizontal lines are close together
    if len(horizontal_lines) >= 2:
        horizontal_lines.sort(key=lambda line: line[1])  # Sort by y-coordinate
        for i in range(len(horizontal_lines) - 1):
            y_diff = abs(horizontal_lines[i][1] - horizontal_lines[i + 1][1])
            if 10 < y_diff < 50:  # Define the expected distance range
                print("STOP")  # Output STOP command
                return "STOP"
    
    return "CONTINUE"

# Test the function with an image
image_path = "C:/Users/ASUS_PC/Downloads/stop_sign_line.jpg"
image = cv2.imread(image_path)
status = detect_stop_symbol(image)
print("Robot Status:", status)

# Display the image (for debugging)
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
