import cv2
import numpy as np
import socket
import struct

# Socket setup
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(("0.0.0.0", 8080))  # Listen on all network interfaces
server_socket.listen(5)

print("Waiting for connection...")

def detect_horizontal_lines(frame, threshold=50, area_threshold=65):
    """
    Function to detect if four boxes in a 2x2 grid contain enough black pixels.
    Includes gaps between rows and columns.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    frame_height, frame_width = frame.shape[:2]
    
    # Define row positions with gap
    box_height = int(frame_height * 0.1)  # 10% of frame height per box
    row_gap = int(frame_height * 0.2)  # 5% gap between rows

    roi_y1_start = int(frame_height * 0.5)  # Start at 50% of height
    roi_y1_end = roi_y1_start + box_height
    roi_y2_start = roi_y1_end + row_gap
    roi_y2_end = roi_y2_start + box_height

    # Define column positions with gap
    box_width = int(frame_width * 0.15)  # 15% of frame width per box
    col_gap = int(frame_width * 0.3)  # 30% gap between columns

    roi_x1_start = int(frame_width * 0.2)  # Start at 20% of frame width
    roi_x1_end = roi_x1_start + box_width
    roi_x2_start = roi_x1_end + col_gap
    roi_x2_end = roi_x2_start + box_width

    # Define four ROIs (2x2 grid with gaps)
    rois = [
        (roi_y1_start, roi_y1_end, roi_x1_start, roi_x1_end),  # Top-Left
        (roi_y1_start, roi_y1_end, roi_x2_start, roi_x2_end),  # Top-Right
        (roi_y2_start, roi_y2_end, roi_x1_start, roi_x1_end),  # Bottom-Left
        (roi_y2_start, roi_y2_end, roi_x2_start, roi_x2_end)   # Bottom-Right
    ]

    black_percentages = []
    for (y1, y2, x1, x2) in rois:
        roi = mask[y1:y2, x1:x2]
        black_pixels = cv2.countNonZero(roi)
        total_pixels = roi.shape[0] * roi.shape[1]
        black_percentage = (black_pixels / total_pixels) * 100
        black_percentages.append(black_percentage)

        # Draw the boxes
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Decision logic:
    top_left, top_right, bottom_left, bottom_right = black_percentages
    bottom_filled = bottom_left >= area_threshold and bottom_right >= area_threshold
    all_filled = top_left >= area_threshold and top_right >= area_threshold and bottom_filled

    if all_filled:
        direction = "STOP"
    elif bottom_filled:
        direction = "PAST_LINE"
    else:
        direction = "FORWARD"

    return direction, black_percentages

def receive_video(conn):
    data = b""
    payload_size = struct.calcsize("Q")

    try:
        while True:
            while len(data) < payload_size:
                packet = conn.recv(4096)
                if not packet:
                    print("Client disconnected.")
                    return
                data += packet

            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("Q", packed_msg_size)[0]

            while len(data) < msg_size:
                data += conn.recv(4096)

            frame_data = data[:msg_size]
            data = data[msg_size:]

            # Decode JPEG frame
            frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)

            # Detect horizontal lines
            direction, black_percentages = detect_horizontal_lines(frame)
            conn.sendall(direction.encode())

            cv2.putText(frame, f"Direction: {direction}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            for i, bp in enumerate(black_percentages):
                cv2.putText(frame, f"Box{i+1}: {bp:.2f}%", (50, 90 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            resized_frame = cv2.resize(frame, (640, 480))
            cv2.imshow("Real-Time Video", resized_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Stopping stream...")
                return
    except (ConnectionResetError, BrokenPipeError):
        print("Connection lost. Waiting for new connection...")
        return

while True:
    conn, addr = server_socket.accept()
    print(f"Connected to {addr}")
    receive_video(conn)
    conn.close()

server_socket.close()
cv2.destroyAllWindows()
