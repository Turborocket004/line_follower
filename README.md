[
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

แปลงภาพจาก RGB → Grayscale 
ใช้ Thresholding เพื่อให้ส่วนที่เป็น สีดำ กลายเป็น สีขาว และพื้นหลังเป็น สีดำ]
-------------------------------------------------------------------------------------------------------------------------------------------------
[
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

ใช้ Contour Detection เพื่อตรวจหาขอบของเส้นสีดำ]
-------------------------------------------------------------------------------------------------------------------------------------------------
[
largest_contour = max(contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(largest_contour)
section_height = h // 4
frame_center = frame.shape[1] // 2

คำนวณหา Bounding Box (พื้นที่ที่ครอบคลุมเส้นดำ)
แบ่งเส้นออกเป็น 4 ส่วน โดยหารความสูงของเส้นด้วย 4]
-------------------------------------------------------------------------------------------------------------------------------------------------
[
for i in range(4):
    section_y = int(y + (i + 0.5) * section_height)
    middle_points.append((frame_center, section_y))

 หาจุดกึ่งกลางของแต่ละส่วนของเส้น]
-------------------------------------------------------------------------------------------------------------------------------------------------
[
 hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

color_ranges = {
    "BLACK": [(0, 0, 0), (180, 255, 50)],
    "RED": [(0, 120, 70), (10, 255, 255)],
    "GREEN": [(35, 40, 40), (85, 255, 255)],
    "BLUE": [(90, 50, 70), (128, 255, 255)]
}

ตรวจจับว่าพื้นที่เส้นมีสีอะไร]
-------------------------------------------------------------------------------------------------------------------------------------------------
[
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

mask	ภาพขาวดำ (Binary Mask)	ใช้แสดงเส้นดำที่ตรวจจับได้
deviations	List ของค่าการเบี่ยงเบน	ใช้คำนวณว่าต้องปรับซ้าย/ขวา
middle_points	List ของจุดกึ่งกลางของเส้น	ใช้ระบุโครงสร้างของเส้น
contour_path	Contour ของเส้นดำ	ใช้วาดเส้นดำบนหน้าจอ
direction	"STRAIGHT", "TURN LEFT", "TURN RIGHT" ฯลฯ	ใช้ควบคุมหุ่นยนต์
deviation_value	ค่าเบี่ยงเบนของเส้นจากกึ่งกลาง	ใช้ปรับทิศทางการเคลื่อนที่
line_color	"BLACK", "RED", "GREEN", "BLUE" 	ใช้ระบุสีของเส้น]
-------------------------------------------------------------------------------------------------------------------------------------------------
