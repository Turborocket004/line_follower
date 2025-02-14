import cv2
import numpy as np

def detect_black_line(frame, threshold=60):
    """
    ตรวจจับเส้นสีดำ คำนวณค่าการเบี่ยงเบนจากจุดกึ่งกลาง และวาดขอบเขตที่สามารถติดตามรูปร่างแบบ L หรือ S ได้
    """
    # แปลงภาพเป็นระดับสีเทา
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ใช้ thresholding เพื่อแยกส่วนที่เป็นสีดำ (กลับค่าเป็นสีขาว)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    # ค้นหาขอบเขตของพื้นที่สีดำทั้งหมด
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    deviations = []  # เก็บค่าการเบี่ยงเบนของ centroid แต่ละส่วน
    middle_points = []  # เก็บจุดกลางของแต่ละส่วน (อ้างอิงตำแหน่งกลางภาพ)
    contour_path = None  # เก็บเส้นขอบที่ตรวจพบ

    if contours:
        # เลือกขอบเขตที่ใหญ่ที่สุด (คิดว่าเป็นเส้นหลัก)
        largest_contour = max(contours, key=cv2.contourArea)

        # ใช้การประมาณรูปทรงเพื่อลดจำนวนจุด ทำให้สามารถติดตามรูปร่างแบบ L หรือ S ได้ดีขึ้น
        epsilon = 0.005 * cv2.arcLength(largest_contour, True)  # ค่าลดลงเพื่อความแม่นยำ
        contour_path = cv2.approxPolyDP(largest_contour, epsilon, True)

        # หาขอบเขตของพื้นที่สีดำและแบ่งออกเป็น 4 ส่วน
        x, y, w, h = cv2.boundingRect(largest_contour)
        section_height = h // 4  # แบ่งออกเป็น 4 ส่วนตามแนวตั้ง

        frame_center = frame.shape[1] // 2  # ตำแหน่งกึ่งกลางของภาพตามแนวแกน X

        # วาดจุดอ้างอิงตรงกลาง (4 จุดที่คงที่)
        for i in range(4):
            section_y = int(y + (i + 0.5) * section_height)  # คำนวณตำแหน่งแนวแกน Y ของแต่ละส่วน
            middle_points.append((frame_center, section_y))  # บันทึกจุดกึ่งกลางแนวตั้ง

            # สร้าง mask เฉพาะส่วนที่เราต้องการตรวจจับ centroid
            section_mask = np.zeros_like(mask)
            section_mask[max(0, section_y - section_height // 2):section_y + section_height // 2, :] = mask[
                max(0, section_y - section_height // 2):section_y + section_height // 2, :]

            # ค้นหา contours ของเส้นสีดำในแต่ละส่วน
            section_contours, _ = cv2.findContours(section_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if section_contours:
                # เลือก contour ที่ใหญ่ที่สุดในส่วนนี้
                section_largest = max(section_contours, key=cv2.contourArea)

                # คำนวณจุดศูนย์กลางของ contour
                M = cv2.moments(section_largest)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])  # คำนวณตำแหน่ง X ของจุดศูนย์กลาง
                    cy = int(M["m01"] / M["m00"])  # คำนวณตำแหน่ง Y ของจุดศูนย์กลาง

                    # คำนวณค่าการเบี่ยงเบนจากจุดกลางภาพ
                    deviation = cx - frame_center
                    deviations.append((cx, cy, deviation))  # เก็บค่าการเบี่ยงเบน

    return mask, deviations, middle_points, contour_path

# เปิดกล้อง
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ตรวจจับเส้นสีดำและรับค่าการเบี่ยงเบน + จุดกลาง
    mask, deviations, middle_points, contour_path = detect_black_line(frame)

    # วาดจุดอ้างอิงกลางภาพ (4 จุดสีแดง)
    for (mx, my) in middle_points:
        cv2.circle(frame, (mx, my), 5, (0, 0, 255), -1)  # จุดสีแดง

    # วาดจุดศูนย์กลางที่ตรวจพบในแต่ละส่วน
    for i, (cx, cy, deviation) in enumerate(deviations):
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)  # จุดสีน้ำเงินแสดงตำแหน่ง centroid
        cv2.putText(frame, f"Dev {i+1}: {deviation}", (50, 50 + i * 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)  # แสดงค่าการเบี่ยงเบนบนภาพ

    # วาดเส้นขอบสีเขียวที่ติดตามเส้นสีดำ (L หรือ S ได้)
    if contour_path is not None:
        cv2.drawContours(frame, [contour_path], -1, (0, 255, 0), 2)  # ขอบเขียวตามเส้นดำ

    # แสดงผลลัพธ์
    cv2.imshow("Original Frame", frame)
    cv2.imshow("Binary Mask", mask)

    # กด 'q' เพื่อออกจากโปรแกรม
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดการเชื่อมต่อกล้องและหน้าต่างทั้งหมด
cap.release()
cv2.destroyAllWindows()
