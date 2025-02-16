import cv2
import numpy as np

def detect_black_line_and_color(frame, threshold=60):
    """
    ฟังก์ชันสำหรับตรวจจับเส้นสีดำ, คำนวณทิศทาง และตรวจจับสีของเส้นที่เป็น (ดำ, แดง, เขียว, น้ำเงิน)
    """
    # แปลงภาพเป็นระดับสีเทา (Grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ใช้ Threshold เพื่อแยกเส้นสีดำออกจากพื้นหลัง
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    # ค้นหา Contours ของเส้นดำ
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    deviations = []  # เก็บค่าการเบี่ยงเบนของเส้น
    middle_points = []  # จุดกึ่งกลางแต่ละส่วนของเส้น
    contour_path = None  # เส้นขอบของเส้นดำ
    direction = "STOP"  # ค่าทิศทางเริ่มต้น
    deviation_value = 0  # ค่าการเบี่ยงเบน
    line_color = "NO COLOR"  # สีของเส้น

    if contours:
        # เลือก Contour ที่ใหญ่ที่สุด สมมติว่าเป็นเส้นหลัก
        largest_contour = max(contours, key=cv2.contourArea)

        # ทำให้เส้นเรียบขึ้น
        epsilon = 0.005 * cv2.arcLength(largest_contour, True)
        contour_path = cv2.approxPolyDP(largest_contour, epsilon, True)

        # สร้าง Mask ของเส้นดำ
        line_mask = np.zeros_like(mask)
        cv2.drawContours(line_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        # แปลงภาพเป็น HSV เพื่อใช้ตรวจจับสี
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # กำหนดช่วงค่าสีสำหรับตรวจจับ
        color_ranges = {
            "BLACK": [(0, 0, 0), (180, 255, 50)],  # สีดำ
            "RED": [(0, 120, 70), (10, 255, 255)],  # สีแดง
            "GREEN": [(35, 40, 40), (85, 255, 255)],  # สีเขียว (ปรับค่าจากเดิม)
            "BLUE": [(90, 50, 70), (128, 255, 255)]  # สีน้ำเงิน
        }

        # ตรวจจับสีของเส้น
        for color, (lower, upper) in color_ranges.items():
            color_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))  # สร้าง Mask สำหรับสี
            color_mask = cv2.bitwise_and(color_mask, color_mask, mask=line_mask)  # ตรวจสอบเฉพาะพื้นที่เส้น

            if cv2.countNonZero(color_mask) > 500:  # ถ้ามีสีมากพอ
                line_color = f"{color} DETECTED"
                break  # พบสีแล้ว หยุดทำงาน

        # คำนวณทิศทางของหุ่นยนต์จากเส้นดำ
        x, y, w, h = cv2.boundingRect(largest_contour)  # หา Bounding Box ของเส้นดำ
        section_height = h // 4  # แบ่งเส้นออกเป็น 4 ส่วน
        frame_center = frame.shape[1] // 2  # กึ่งกลางของเฟรม

        top_dot = None
        for i in range(4):  # วนลูปตรวจจับจุดกลางของแต่ละส่วน
            section_y = int(y + (i + 0.5) * section_height)  # คำนวณตำแหน่งแกน Y
            middle_points.append((frame_center, section_y))

            # สร้าง Mask เฉพาะบริเวณที่ต้องการ
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

                    if i == 0:  # จุดบนสุดของเส้น
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

# เปิดกล้อง
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ตรวจจับเส้นดำ + คำนวณทิศทาง + ตรวจจับสีของเส้น
    mask, deviations, middle_points, contour_path, direction, deviation_value, line_color = detect_black_line_and_color(frame)

    # วาดจุดกลางของแต่ละส่วนบนเฟรม
    for (mx, my) in middle_points:
        cv2.circle(frame, (mx, my), 5, (0, 0, 255), -1)

    for (cx, cy, deviation) in deviations:
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
        cv2.putText(frame, f"Dev: {deviation}", (cx, cy-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    if contour_path is not None:
        cv2.drawContours(frame, [contour_path], -1, (0, 255, 0), 2)

    # แสดงข้อมูลทิศทาง และสีของเส้นบนเฟรม
    cv2.putText(frame, f"Direction: {direction}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Deviation: {deviation_value}", (50, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f"Line Color: {line_color}", (50, 130), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # แสดงผล
    cv2.imshow("Original Frame", frame)
    cv2.imshow("Binary Mask", mask)

    # กด 'q' เพื่อออกจากโปรแกรม
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
