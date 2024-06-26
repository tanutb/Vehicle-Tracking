# Vehicle Tracking with YOLOv8 + SORT
<p align="center">
  <img src="https://github.com/tanutb/Vehicle-Tracking/blob/main/content/tracking.gif" >
</p>

## Overview
This project focuses on tracking vehicles on the road using the YOLOv8 model combined with SORT (Simple Online Real-time Tracking) to enhance detection performance. Despite the implementation, there are some performance issues and bugs that need to be addressed.

## Dataset
We use the dataset provided by <a href=https://github.com/MaryamBoneh/Vehicle-Detection> MaryamBoneh/Vehicle-Detection</a> to train YOLOv8.

## Installation and Usage
### Installation
To get started with the project, you need to install the following dependencies:

- Ultralytics YOLOv8: This is the main object detection model.
- OpenCV: For image and video processing.
- Filterpy: For the implementation of the Kalman Filter used in SORT.
  
You can install these dependencies using pip:

```
pip install ultralytics opencv-python filterpy
```

Running the Tracker
To run the vehicle tracking script, execute the following command:
```
python track.py
```

## Limitations
In the current implementation, there are some limitations that may affect performance:

- <b>Object Detection Accuracy</b>: The current implementation sometimes detects incorrect objects, leading to false positives.

<p align="center">
  <img src="https://github.com/tanutb/Vehicle-Tracking/blob/main/content/wrong_prediction.png">
</p>

- <b>Tracking Multiple Vehicles</b>: The tracking struggles with detecting and tracking a large number of vehicles simultaneously, which affects overall performance.

- <b>Tracking Ghost</b>: The tracker may occasionally follow the wrong, leading to inaccurate tracking results.

<p align="center">
  <img src="https://github.com/tanutb/Vehicle-Tracking/blob/main/content/wrong_prediction2.png">
</p>
