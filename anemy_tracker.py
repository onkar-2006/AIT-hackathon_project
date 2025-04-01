import cv2
import torch
import numpy as np
from twilio.rest import Client  # Import Twilio SDK

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load YOLOv5 small model

video_stream = cv2.VideoCapture("9466311-hd_1920_1080_25fps.mp4")  # Replace with your video file path

if not video_stream.isOpened():
    print("Error: Could not open video stream.")
    exit()

ret, frame = video_stream.read()
if not ret:
    print("Error: Could not read first frame.")
    exit()

frame_width = frame.shape[1]
frame_height = frame.shape[0]

resize_width = 800
resize_height = int((resize_width / frame_width) * frame_height)

def classify_color(frame, bbox):
    x1, y1, x2, y2 = bbox
    roi = frame[int(y1):int(y2), int(x1):int(x2)]

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    army_lower = np.array([35, 50, 50])  
    army_upper = np.array([85, 255, 255])  

    army_mask = cv2.inRange(hsv_roi, army_lower, army_upper)

    army_percentage = np.sum(army_mask) / (roi.shape[0] * roi.shape[1])

    if army_percentage > 0.3: 
        return "Army"
    else:
        return "Enemy"  

def send_sms(to_phone, message):
    account_sid = 'ACb62ce40a0ea94e1d9a0057f231fa1c8f'
    auth_token = '32207013a38f1f3ab3736c8b5a5621d4'
    from_phone = '+12542805945'

    client = Client(account_sid, auth_token)

    message = client.messages.create(
        body=message,
        from_=from_phone,
        to=to_phone
    )

    print(f"Message sent: {message.sid}")

enemy_count = 0
headquarters_phone = "+91 86258 35294"  

message_sent = False

processed_bboxes = []

iou_threshold = 0.3

def compute_iou(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    xi1 = max(x1, x1_2)
    yi1 = max(y1, y1_2)
    xi2 = min(x2, x2_2)
    yi2 = min(y2, y2_2)

    intersection_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    bbox1_area = (x2 - x1) * (y2 - y1)
    bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

    union_area = bbox1_area + bbox2_area - intersection_area

    iou = intersection_area / union_area
    return iou

while True:
    ret, frame = video_stream.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (resize_width, resize_height))

    results = model(frame_resized)  
    detections = results.xyxy[0].cpu().numpy()  

    frame_enemy_count = 0

    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        if int(cls) == 0 and conf > 0.3:  
            classification = classify_color(frame_resized, (x1, y1, x2, y2))

            is_duplicate = False
            for processed_bbox in processed_bboxes:
                if compute_iou((x1, y1, x2, y2), processed_bbox) > iou_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                processed_bboxes.append((x1, y1, x2, y2))  

                label = f'{classification} ({conf:.2f})'
                cv2.rectangle(frame_resized, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame_resized, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                if classification == "Enemy":
                    frame_enemy_count += 1
                    if not message_sent:  # Send message only once
                        message = "An enemy soldier has been detected in the video stream."
                        send_sms(headquarters_phone, message)
                        message_sent = True  

    enemy_count += frame_enemy_count

    cv2.putText(frame_resized, f'Total Enemies: {enemy_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Object Detection with Color Classification", frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_stream.release()
cv2.destroyAllWindows()
