# EdgeVision Traffic Analytics
A real-time edge computer vision application built with YOLOv8 and OpenCV to detect, track, and count road vehicles while logging structured traffic analytics into CSV format.

## Features
- **Real-Time Object Detection**: Uses YOLOv8 for accurate identification of cars, buses, trucks, and motorcycles.
- **Line-Crossing Logic**: Increments counters dynamically as tracked objects cross designated spatial lines.
- **Automated Logging**: Continuous structured data persistence in `traffic_log.csv`.
- **Live Visual Overlay**: On-screen dynamic metric cards displaying live counts.
## Tech Stack
- Python 3.x
- Ultralytics (YOLOv8)
- OpenCV
- NumPy
## Installation & Setup
1. **Clone the Repository**
   ```bash
   git clone [https://github.com/Komal53131/Edge-Vision-Traffic-Analytics.git](https://github.com/Komal53131/Edge-Vision-Traffic-Analytics.git)
   cd Edge-Vision-Traffic-Analytics
