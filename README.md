# Enemy Detection and Tracking System Using OpenCV for Indian Army Safety

## Table of Contents
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Installation Instructions](#installation-instructions)
- [Usage](#usage)
- [System Architecture](#system-architecture)
- [Results](#results)
- [Future Scope](#future-scope)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
This project focuses on **enemy detection** and **tracking** using **OpenCV** to ensure **real-time safety** for the Indian Army. The system uses video feeds (from surveillance cameras or drones) to detect enemy threats and track their movements. It provides **automated alerts** to military personnel to help in timely decision-making and responses.

## Technologies Used
- **OpenCV**: For image processing and computer vision tasks (detection and tracking).
- **Python**: The main programming language used to develop the system.
- **Twilio**: To send automated **SMS alerts** to designated personnel for real-time updates.
- **Deep Learning Models**: (Optional) For advanced object detection (e.g., YOLOv5 for detecting specific targets like vehicles, soldiers, etc.).
- **Surveillance Systems**: Integrated with live video feeds from cameras or drones.

## Installation Instructions

### Prerequisites:
Before you begin, ensure you have the following installed:
- Python 3.x
- `pip` (Python package manager)
- Virtual environment (optional but recommended)

## System Architecture

Input: Video feed from a camera or a pre-recorded video.

Detection: The video feed is processed frame by frame, and YOLOv5 is used to detect objects in each frame.

Tracking: After detecting an object, a tracking algorithm is used to follow its movement.

Alert: If an enemy is detected, an SMS alert is sent to the specified number via Twilio.

Output: A real-time video feed with bounding boxes showing detected threats, and the corresponding SMS alert.

## Results
The system was able to:
# Object Detection Project

This project uses YOLOv5 for object detection in video streams. Below is an example of the detection:

<img src="https://github.com/onkar-2006/AIT-hackathon_project/blob/main/alert%20message.jpg" alt="Image Description" width="300" />

Detect enemies (people, vehicles, etc.) from the video feed in real-time.

Track the movement of the detected objects.

Send SMS alerts when an enemy is detected, notifying the headquarters or relevant personnel.

## Sample Output:
Detected Enemy (Real-Time Video Feed):

The bounding boxes are drawn around the detected person/vehicle.

SMS Alert:

Upon detecting an enemy, a message is sent: Alert: Enemy detected at [location], please take necessary action.


