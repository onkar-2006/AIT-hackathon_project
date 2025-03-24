import cv2
import torch
import numpy as np
from twilio.rest import Client  # Import Twilio SDK
import os

# Load YOLOv5 model for object detection (soldiers, enemies)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load YOLOv5 small model

# Initialize video stream
video_stream = cv2.VideoCapture("9466311-hd_1920_1080_25fps.mp4")  # Replace with your video file path

# Check if video is opened correctly
if not video_stream.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Get original frame dimensions
ret, frame = video_stream.read()
if not ret:
    print("Error: Could not read first frame.")
    exit()

frame_width = frame.shape[1]
frame_height = frame.shape[0]

# Resize the frame for better display
resize_width = 800  # You can adjust this based on your screen size
resize_height = int((resize_width / frame_width) * frame_height)  # Maintain aspect ratio

# Function to classify based on color (green for army, everything else for enemy)
def classify_color(frame, bbox):
    x1, y1, x2, y2 = bbox
    # Crop the region of interest (ROI) from the frame
    roi = frame[int(y1):int(y2), int(x1):int(x2)]

    # Convert the ROI to HSV color space (better for color-based analysis)
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Define the color range for army (green color)
    army_lower = np.array([30, 50, 50])  # Lower bound of green
    army_upper = np.array([80, 255, 255])  # Upper bound of green

    # Mask for the army (green) color from the ROI
    army_mask = cv2.inRange(hsv_roi, army_lower, army_upper)

    # Calculate the percentage of the ROI covered by green (army) color
    army_percentage = np.sum(army_mask) / (roi.shape[0] * roi.shape[1])

    # If the army color occupies more area, classify as soldier (army), else enemy
    if army_percentage > 0.6:  # You can adjust this threshold if needed
        return "Army"
    else:
        return "Enemy"

# Function to send SMS notification using Twilio
def send_sms(to_phone, message):
    # Twilio credentials (Account SID and Auth Token from your Twilio Console)
    account_sid = 'ACb62ce40a0ea94e1d9a0057f231fa1c8f'  # Replace with your actual SID
    auth_token = 'e690ae51eaee6b4227b0834aadf3937f'    # Replace with your actual Auth Token
    from_phone = '+12542805945'  # Replace with your Twilio phone number

    # Create a Twilio client
    client = Client(account_sid, auth_token)

    # Send an SMS message
    message = client.messages.create(
        body=message,
        from_=from_phone,  # Your Twilio phone number
        to=to_phone  # The recipient's phone number
    )

    print(f"Message sent: {message.sid}")

# Initialize counter for enemy count
enemy_count = 0

# Define the recipient's phone number (headquarters phone number)
headquarters_phone = "+91 86258 35294"  # Replace with the headquarters phone number

# Loop through each frame and classify the bounding boxes
while True:
    ret, frame = video_stream.read()
    if not ret:
        break

    # Resize frame for display while maintaining the aspect ratio
    frame_resized = cv2.resize(frame, (resize_width, resize_height))

    # Perform object detection to find persons (soldiers/enemies)
    results = model(frame_resized)
    detections = results.xyxy[0].cpu().numpy()

    # Reset enemy count for each frame
    frame_enemy_count = 0

    # Iterate over each detection (filtered by class 'person')
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        if int(cls) == 0 and conf > 0.5:  # Class 'person' and confidence > 0.5
            # Classify based on color (green for army, other colors for enemy)
            classification = classify_color(frame_resized, (x1, y1, x2, y2))

            # Draw the bounding box and the classification label
            label = f'{classification} ({conf:.2f})'
            cv2.rectangle(frame_resized, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame_resized, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Count enemies
            if classification == "Enemy":
                frame_enemy_count += 1

                # Send SMS notification when an enemy is detected (ensure no spam)
                message = "An enemy soldier has been detected in the video stream."
                send_sms(headquarters_phone, message)

    # Update the total enemy count
    enemy_count += frame_enemy_count

    # Display the total number of enemies detected on the screen
    cv2.putText(frame_resized, f'Total Enemies: {enemy_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame with classification labels and enemy count
    cv2.imshow("Object Detection with Color Classification", frame_resized)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video stream and close windows
video_stream.release()
cv2.destroyAllWindows()
