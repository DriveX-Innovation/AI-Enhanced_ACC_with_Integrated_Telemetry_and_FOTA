# Use your exact Python version
FROM python:3.11.8-slim

# Install system dependencies for OpenCV 
# 2. Install system dependencies for OpenCV AND X11/XCB for display
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1-mesa-glx \
    # The absolute essentials for XCB/Qt
    libxcb-xinerama0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xfixes0 \
    libxcb-xkb1 \
    libxkbcommon-x11-0 \
    libdbus-1-3 \
    # Font and Render support
    libfontconfig1 \
    libxrender1 \
    libxi6 \
    && rm -rf /var/lib/apt/lists/*
# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project folder
COPY . .

# Adjust the path to your main file
# Example: If your file is in 'scripts/main_detect.py'
CMD ["python", "/app/TrafficLaneDetector/videoDetection.py"]