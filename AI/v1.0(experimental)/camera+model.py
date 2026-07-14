import cv2
import time
from threading import Thread
from ultralytics import YOLO
from ObjectDetector.distanceMeasure import SingleCamDistanceMeasure

# ==========================
# ABSOLUTE CONFIGURATION
# ==========================
MODEL_PATH = "/home/drivx/adas-sys/Vehicle-CV-ADAS-master/models/yolov8n_ncnn_model"
WEBCAM_INDEX = 0  # Switched from file to local USB webcam

class ThreadedWebcamStream:
    """
    Decouples camera frame grabbing from the main execution thread.
    Prevents the model from pausing while waiting for the USB hardware sync.
    """
    def __init__(self, src=WEBCAM_INDEX):
        # Force V4L2 backend for native Linux efficiency
        self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        
        # OPTIMIZATION 1: Hardware-level frame sizing
        # Setting camera output close to your model's 640x640 inference size 
        # completely eliminates heavy CPU image-resizing math later.
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Request compressed MJPG stream from webcam to maximize USB bus bandwidth
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        self.ret, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        # Run thread as a daemon so it dies cleanly when the main script ends
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stop()
                return
            self.ret = ret
            self.frame = frame

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()


def main():
    print("Loading YOLOv8 NCNN Model...")
    # Initialize the NCNN model natively
    model = YOLO(MODEL_PATH, task='detect')
    classes = [model.names[i] for i in range(len(model.names))]

    print("Initializing Optimized Webcam Stream...")
    vs = ThreadedWebcamStream(src=WEBCAM_INDEX).start()
    time.sleep(1.0)  # Allow webcam sensor warm-up time

    distanceDetector = SingleCamDistanceMeasure()

    fps = 0
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = vs.read()
        if not ret or frame is None:
            print("Waiting for webcam frames...")
            continue

        # OPTIMIZATION 2: Stream=True reduces generator memory allocations
        results = model.predict(frame, imgsz=640, conf=0.25, verbose=False, stream=True, classes=[2, 3, 5, 7])
        
        for result in results:
            detections = []
            boxes = result.boxes
            
            # OPTIMIZATION 3: Vectorized extraction via NumPy array slicing
            # Replaces slow item-by-item Python loops over individual Box tensors
            if len(boxes) > 0:
                xyxy = boxes.xyxy.numpy().astype(int)
                confs = boxes.conf.numpy()
                clss = boxes.cls.numpy().astype(int)
                
                # Zip structures natively for the distance evaluation class
                detections = [
                    (xyxy[i][0], xyxy[i][1], xyxy[i][2], xyxy[i][3], confs[i], clss[i])
                    for i in range(len(boxes))
                ]

            # Update ADAS Logic
            distanceDetector.updateDistance(detections, classes)

            # Draw bounding boxes automatically using Ultralytics
            annotated_frame = result.plot()

            # Draw distance overlays
            distanceDetector.DrawDetectedOnFrame(annotated_frame)

            # FPS Calculation
            frame_count += 1
            if frame_count >= 30:
                fps = 30 / (time.time() - start_time)
                frame_count = 0
                start_time = time.time()

            cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # OPTIMIZATION 4: VNC Bottleneck Mitigation
            # Rendering GUI frames over a VNC link adds network-bound lag.
            cv2.imshow("ADAS System - NCNN", annotated_frame)
            
        if cv2.waitKey(1) == 27:  # Press 'ESC' to exit cleanly
            break

    vs.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
