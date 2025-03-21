def detect_colors(frame, threshold=50, area_threshold=65):
    """Detects black, red, green, and blue colors in four ROIs."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, black_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    
    # Color masks in HSV
    red_mask = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255)) + cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
    green_mask = cv2.inRange(hsv, (40, 40, 40), (90, 255, 255))
    blue_mask = cv2.inRange(hsv, (90, 50, 50), (140, 255, 255))
    
    frame_height, frame_width = frame.shape[:2]
    box_height = int(frame_height * 0.1)
    row_gap = int(frame_height * 0.2)
    box_width = int(frame_width * 0.15)
    col_gap = int(frame_width * 0.3)
    
    roi_y1_start = int(frame_height * 0.5)
    roi_y1_end = roi_y1_start + box_height
    roi_y2_start = roi_y1_end + row_gap
    roi_y2_end = roi_y2_start + box_height
    
    roi_x1_start = int(frame_width * 0.2)
    roi_x1_end = roi_x1_start + box_width
    roi_x2_start = roi_x1_end + col_gap
    roi_x2_end = roi_x2_start + box_width
    
    rois = [
        (roi_y1_start, roi_y1_end, roi_x1_start, roi_x1_end),  # Top-Left
        (roi_y1_start, roi_y1_end, roi_x2_start, roi_x2_end),  # Top-Right
        (roi_y2_start, roi_y2_end, roi_x1_start, roi_x1_end),  # Bottom-Left
        (roi_y2_start, roi_y2_end, roi_x2_start, roi_x2_end)   # Bottom-Right
    ]
    
    results = []
    
    for (y1, y2, x1, x2) in rois:
        roi_black = black_mask[y1:y2, x1:x2]
        roi_red = red_mask[y1:y2, x1:x2]
        roi_green = green_mask[y1:y2, x1:x2]
        roi_blue = blue_mask[y1:y2, x1:x2]
        
        total_pixels = roi_black.shape[0] * roi_black.shape[1]
        black_percentage = (cv2.countNonZero(roi_black) / total_pixels) * 100
        red_percentage = (cv2.countNonZero(roi_red) / total_pixels) * 100
        green_percentage = (cv2.countNonZero(roi_green) / total_pixels) * 100
        blue_percentage = (cv2.countNonZero(roi_blue) / total_pixels) * 100
        
        results.append((black_percentage, red_percentage, green_percentage, blue_percentage))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # Check for color detection across all boxes
    color_detected = {
        'black': all(r[0] >= area_threshold for r in results),
        'red': all(r[1] >= area_threshold for r in results),
        'green': all(r[2] >= area_threshold for r in results),
        'blue': all(r[3] >= area_threshold for r in results),
    }
    
    if any(color_detected.values()):  # If any color is detected in all boxes
        return "STOP", results
    
    # Check for detection in individual boxes
    detected_colors = []
    for i, (black, red, green, blue) in enumerate(results):
        if red >= area_threshold:
            detected_colors.append(f"Box{i+1}: RED")
        elif green >= area_threshold:
            detected_colors.append(f"Box{i+1}: GREEN")
        elif blue >= area_threshold:
            detected_colors.append(f"Box{i+1}: BLUE")
        elif black >= area_threshold:
            detected_colors.append(f"Box{i+1}: BLACK")
    
    return ", ".join(detected_colors) if detected_colors else "FORWARD", results
