import cv2, time
import numpy as np


from taskConditions import TaskConditions
from ObjectDetector import yoloDetector
from ObjectDetector.utils import ObjectModelType
from ObjectDetector.distanceMeasure import SingleCamDistanceMeasure




video_path = "./temp/test.mp4"

object_config = {
	"model_path": '/home/drivx/adas-sys/Vehicle-CV-ADAS-master/models/yolov8n.onnx',
	"model_type" : ObjectModelType.YOLOV8,
	"classes_path" : '/home/drivx/adas-sys/Vehicle-CV-ADAS-master/models/coco_label.txt',
	"box_score" : 0.4,
	"box_nms_iou" : 0.5
}
class ControlPanel(object):
	def __init__(self):
		self.fps = 0
		self.frame_count = 0
		self.start = time.time()

		self.curve_status = None

	def _updateFPS(self):
		"""
		Update FPS.

		Args:
			None

		Returns:
			None
		"""
		self.frame_count += 1
		if self.frame_count >= 30:
			self.end = time.time()
			self.fps = self.frame_count / (self.end - self.start)
			self.frame_count = 0
			self.start = time.time()

if __name__ == "__main__":

	# Initialize read and save video 
	cap = cv2.VideoCapture(video_path)
	if (not cap.isOpened()) :
		raise Exception("video path is error. please check it.")
	width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
	vout = cv2.VideoWriter(video_path[:-4]+'_out.mp4', fourcc , 30.0, (width, height))
	cv2.namedWindow("ADAS Simulation", cv2.WINDOW_NORMAL)	
	
	# object detection model
	
	if ( ObjectModelType.EfficientDet == object_config["model_type"]):
		
	
		yoloDetector.set_defaults(object_config)
		objectDetector = yoloDetector
	distanceDetector = SingleCamDistanceMeasure()
	

    	# display panel
	displayPanel = ControlPanel()
	analyzeMsg = TaskConditions()
	while cap.isOpened():

		ret, frame = cap.read() # Read frame from the video
		if ret:	
			frame_show = frame.copy()


			#========================== Detect Model =========================
			obect_time = time.time()
			objectDetector.DetectFrame(frame)
			obect_infer_time = round(time.time() - obect_time, 2)

			#========================= Analyze Status ========================
			distanceDetector.updateDistance(objectDetector.object_info)
			
            
			#========================== Draw Results =========================
			
			
			
			
			objectDetector.DrawDetectedOnFrame(frame_show)
			
			distanceDetector.DrawDetectedOnFrame(frame_show)

			
	
			
			cv2.imshow("ADAS Simulation", frame_show)

		else:
			break
		vout.write(frame_show)	
		if cv2.waitKey(1) == ord('q'): # Press key q to stop
			break

	vout.release()
	cap.release()
	cv2.destroyAllWindows()
			

			
			

			

