import sys
carla_egg = r"D:\graduation project related\Vehicle-CV-ADAS-master\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.11-py3.7-win-amd64.egg"

if carla_egg not in sys.path:
    sys.path.append(carla_egg)

import carla
import random
import time
import math
import numpy as np
import cv2
import os

# ===================== PATHS =====================
MODEL_PATH = "models/yolov8n.onnx"
LABELS_PATH = "models/coco_label.txt"

# ===================== LOAD COCO LABELS =====================
with open(LABELS_PATH, "r") as f:
    COCO_CLASSES = [line.strip() for line in f.readlines()]

print("✅ COCO labels loaded:", len(COCO_CLASSES))


# ===================== YOLOv8 ONNX DETECTOR =====================
class YOLOv8Detector:
    def __init__(self, model_path, conf_threshold=0.5):
        self.conf_threshold = conf_threshold
        self.input_width = 640
        self.input_height = 480

        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        print("✅ YOLOv8 ONNX model loaded (480x640) on CPU")

    def detect(self, image):
        h, w = image.shape[:2]

        blob = cv2.dnn.blobFromImage(
            image,
            1/255.0,
            (self.input_width, self.input_height),
            swapRB=True,
            crop=False
        )
        self.net.setInput(blob)
        outputs = self.net.forward()

        predictions = outputs[0].transpose()

        scores_matrix = predictions[:, 4:]
        boxes_matrix = predictions[:, :4]

        class_ids = np.argmax(scores_matrix, axis=1)
        max_scores = np.max(scores_matrix, axis=1)

        mask = max_scores >= self.conf_threshold
        filtered_boxes = boxes_matrix[mask]
        filtered_scores = max_scores[mask]
        filtered_class_ids = class_ids[mask]

        x_scale = w / self.input_width
        y_scale = h / self.input_height

        final_boxes = []
        final_scores = []
        final_class_ids = []

        for box, score, class_id in zip(filtered_boxes, filtered_scores, filtered_class_ids):
            cx, cy, bw, bh = box

            cx *= x_scale
            cy *= y_scale
            bw *= x_scale
            bh *= y_scale

            x = int(cx - (bw / 2))
            y = int(cy - (bh / 2))
            w_box = int(bw)
            h_box = int(bh)

            final_boxes.append([x, y, w_box, h_box])
            final_scores.append(float(score))
            final_class_ids.append(int(class_id))

        indices = cv2.dnn.NMSBoxes(final_boxes, final_scores, self.conf_threshold, 0.45)

        return final_boxes, final_scores, final_class_ids, indices


# ===================== CARLA INITIALIZATION =====================
class CarlaInit:
    def __init__(self, port=2000):
        self.client = carla.Client("localhost", port)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()
        self.bp_lib = self.world.get_blueprint_library()
        self.spawn_points = self.world.get_map().get_spawn_points()
        print("✅ Connected to CARLA")

    def spawn_actors(self, car_name="vehicle.audi.a2", num_actors=70):
        ego_bp = self.bp_lib.find(car_name)
        self.ego_vehicle = self.world.try_spawn_actor(
            ego_bp, random.choice(self.spawn_points)
        )

        for _ in range(num_actors):
            vehicle_bp = random.choice(self.bp_lib.filter("vehicle"))
            self.world.try_spawn_actor(vehicle_bp, random.choice(self.spawn_points))

        print("✅ Vehicles spawned")

    def spawn_camera(self):
        cam_bp = self.bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", "640")
        cam_bp.set_attribute("image_size_y", "480")
        cam_bp.set_attribute("fov", "90")

        transform = carla.Transform(carla.Location(x=1.5, z=2.2))
        self.camera = self.world.spawn_actor(
            cam_bp, transform, attach_to=self.ego_vehicle
        )

        print("✅ RGB camera spawned")

    # ✅ NEW: Disable all traffic lights
    def disable_traffic_lights(self):
        traffic_lights = self.world.get_actors().filter("*traffic_light*")

        for tl in traffic_lights:
            tl.set_state(carla.TrafficLightState.Green)
            tl.freeze(True)

        print("✅ All traffic lights set to GREEN and frozen")

    def callback(self, image):
        global frame
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        frame = array.reshape((image.height, image.width, 4)).copy()

    def listen_camera(self):
        self.camera.listen(lambda image: self.callback(image))

    def set_autopilot(self):
        for actor in self.world.get_actors().filter("*vehicle*"):
            actor.set_autopilot(True)
        print("✅ Autopilot enabled")


# ===================== MAIN DETECTION LOOP =====================
class CarlaYOLO:
    def __init__(self):
        self.detector = YOLOv8Detector(MODEL_PATH)
        self.carla = CarlaInit()

    def run(self):
        global frame
        frame = None

        self.carla.spawn_actors()
        self.carla.spawn_camera()
        self.carla.listen_camera()
        self.carla.set_autopilot()

        # ✅ Disable traffic lights here
        self.carla.disable_traffic_lights()

        print("🚀 YOLOv8 + CARLA started (press Q to quit)")

        while True:
            if frame is None:
                continue

            img = frame[:, :, :3].copy()

            boxes, scores, class_ids, indices = self.detector.detect(img)

            if len(indices) > 0:
                for i in indices.flatten():
                    class_id = int(class_ids[i])
                    class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else "Unknown"

                    x, y, w, h = boxes[i]
                    label = f"{class_name}: {scores[i]:.2f}"

                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(img, label, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("CARLA YOLOv8 Detection", img)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()


# ===================== RUN =====================
if __name__ == "__main__":
    CarlaYOLO().run()















'''"""
#####################################
# CONFIGURATION
#####################################
TARGET_SPEED = 15.0      # m/s (~54 km/h)
TIME_GAP = 1.5           # seconds
MIN_DISTANCE = 5.0       # meters

# PID parameters (tuned for CARLA)
KP = 0.8
KI = 0.02
KD = 0.25

#####################################
# PID CONTROLLER CLASS
#####################################
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output


#####################################
# CONNECT TO CARLA
#####################################
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.get_world()
blueprints = world.get_blueprint_library()

#####################################
# SPAWN EGO VEHICLE
#####################################
vehicle_bp = blueprints.find("vehicle.tesla.model3")
spawn_points = world.get_map().get_spawn_points()

ego_vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
if ego_vehicle is None:
    raise RuntimeError("Failed to spawn ego vehicle")

print("Ego vehicle spawned:", ego_vehicle.id)

#####################################
# SPAWN TRAFFIC VEHICLES
#####################################
for _ in range(20):
    bp = random.choice(blueprints.filter("vehicle.*"))
    sp = random.choice(spawn_points)
    v = world.try_spawn_actor(bp, sp)
    if v:
        v.set_autopilot(True)

#####################################
# ATTACH CAMERA
#####################################
camera_bp = blueprints.find("sensor.camera.rgb")
camera_bp.set_attribute("image_size_x", "800")
camera_bp.set_attribute("image_size_y", "600")
camera_bp.set_attribute("fov", "90")

camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)

frame = None

def camera_callback(image):
    global frame
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    frame = array[:, :, :3]

camera.listen(camera_callback)

#####################################
# LOAD YOLOv8n
#####################################
model = ("models/yolov8n.onnx")  # path to your YOLOv8n ONNX model
CLASSES_PATH = "models/coco_label.txt"
CONF_THRES = 0.25
IOU_THRES = 0.45
INPUT_SIZE = (480, 640)  # YOLOv8 default

#####################################
# FIND FRONT VEHICLE USING YOLO + CARLA
#####################################
def get_front_vehicle():
    vehicles = world.get_actors().filter("vehicle.*")
    ego_loc = ego_vehicle.get_location()
    ego_forward = ego_vehicle.get_transform().get_forward_vector()

    front_vehicle = None
    min_distance = 9999

    for v in vehicles:
        if v.id == ego_vehicle.id:
            continue

        loc = v.get_location()
        dx = loc.x - ego_loc.x
        dy = loc.y - ego_loc.y

        forward_dot = dx * ego_forward.x + dy * ego_forward.y
        if forward_dot > 0:  # vehicle in front
            distance = ego_loc.distance(loc)
            if distance < min_distance:
                min_distance = distance
                front_vehicle = v

    return front_vehicle, min_distance


#####################################
# PID CONTROLLER
#####################################
pid = PIDController(KP, KI, KD)
last_time = time.time()

#####################################
# MAIN LOOP
#####################################
while True:
    if frame is None:
        continue

    # Ego speed
    vel = ego_vehicle.get_velocity()
    speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

    # Find front vehicle
    front_vehicle, distance = get_front_vehicle()

    # Safe distance
    safe_distance = speed * TIME_GAP + MIN_DISTANCE

    # Distance error
    if front_vehicle is not None:
        error = safe_distance - distance
    else:
        error = -TARGET_SPEED  # no car ahead → accelerate to target speed

    # PID computation
    now = time.time()
    dt = now - last_time
    last_time = now

    control_signal = pid.compute(error, dt)

    # Convert PID output to throttle/brake
    throttle = 0.0
    brake = 0.0

    if control_signal > 0:
        brake = min(control_signal, 1.0)
    else:
        throttle = min(-control_signal, 1.0)

    control = carla.VehicleControl(throttle=throttle, brake=brake)
    ego_vehicle.apply_control(control)

    # Visualization
    text = f"Speed: {speed:.1f} m/s | Distance: {distance:.1f} m | Safe: {safe_distance:.1f} m"
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("ACC + YOLOv8n (CARLA)", frame)
    cv2.waitKey(1)

    time.sleep(0.05)
'''
