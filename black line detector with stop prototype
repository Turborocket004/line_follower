import cv2
import numpy as np

class BlackLineDetector:
    def __init__(self, threshold=60):
        self.threshold = threshold
    
    def detect_black_line_and_color(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        deviations = []
        middle_points = []
        contour_path = None
        direction = "STOP"
        deviation_value = 0
        line_color = "NO COLOR"

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            epsilon = 0.005 * cv2.arcLength(largest_contour, True)
            contour_path = cv2.approxPolyDP(largest_contour, epsilon, True)

            line_mask = np.zeros_like(mask)
            cv2.drawContours(line_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            color_ranges = {
                "BLACK": [(0, 0, 0), (180, 255, 50)],
                "RED": [(0, 120, 70), (10, 255, 255)],
                "GREEN": [(35, 40, 40), (85, 255, 255)],
                "BLUE": [(90, 50, 70), (128, 255, 255)]
            }

            for color, (lower, upper) in color_ranges.items():
                color_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                color_mask = cv2.bitwise_and(color_mask, color_mask, mask=line_mask)

                if cv2.countNonZero(color_mask) > 500:
                    line_color = f"{color} DETECTED"
                    break

            x, y, w, h = cv2.boundingRect(largest_contour)
            section_height = h // 4
            frame_center = frame.shape[1] // 2
            top_dot = None

            for i in range(4):
                section_y = int(y + (i + 0.5) * section_height)
                middle_points.append((frame_center, section_y))
                
                section_mask = np.zeros_like(mask)
                section_mask[max(0, section_y - section_height // 2):section_y + section_height // 2, :] = mask[
                    max(0, section_y - section_height // 2):section_y + section_height // 2, :]
                
                section_contours, _ = cv2.findContours(section_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if section_contours:
                    section_largest = max(section_contours, key=cv2.contourArea)
                    M = cv2.moments(section_largest)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        deviation = cx - frame_center
                        deviations.append((cx, cy, deviation))
                        
                        if i == 0:
                            top_dot = (cx, cy)

            if deviations:
                deviation_value = deviations[-1][2]
                if deviation_value < -20:
                    direction = "ADJUST RIGHT"
                elif deviation_value > 20:
                    direction = "ADJUST LEFT"
                else:
                    direction = "FORWARD"

            if top_dot:
                deviation = top_dot[0] - frame_center
                if deviation < -100:
                    direction = "TURN RIGHT"
                elif deviation > 100:
                    direction = "TURN LEFT"
                elif deviation < -20:
                    direction = "ADJUST RIGHT"
                elif deviation > 20:
                    direction = "ADJUST LEFT"
                else:
                    direction = "STRAIGHT"

        return mask, deviations, middle_points, contour_path, direction, deviation_value, line_color

class StopSymbolDetector(BlackLineDetector):
    def detect_stop_symbol(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
        
        horizontal_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(y2 - y1) < 10:
                    horizontal_lines.append((x1, y1, x2, y2))

        if len(horizontal_lines) >= 2:
            horizontal_lines.sort(key=lambda line: line[1])
            for i in range(len(horizontal_lines) - 1):
                y_diff = abs(horizontal_lines[i][1] - horizontal_lines[i + 1][1])
                if 10 < y_diff < 50:
                    return "STOP"
        
        return "CONTINUE"

# Example Usage
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detector = StopSymbolDetector()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        mask, deviations, middle_points, contour_path, direction, deviation_value, line_color = detector.detect_black_line_and_color(frame)
        stop_status = detector.detect_stop_symbol(frame)
        
        cv2.putText(frame, f"Direction: {direction}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Deviation: {deviation_value}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Line Color: {line_color}", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Stop Status: {stop_status}", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("Frame", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
