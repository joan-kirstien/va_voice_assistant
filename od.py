from ultralytics import YOLO
import cv2
import math

class ObjectDetector:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)
        self.cap.set(4, 480)
     
        
        self.model = YOLO("yolo-Weights/yolov8n.pt")
        self.classNames = ["person", "bicycle", "car", "wallet", "keys", "bus", "train", "truck", "boat",
                           "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                           "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                           "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                           "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                           "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                           "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                           "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cellphone",
                           "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                           "teddy bear", "hair drier", "toothbrush"]
        

    def detect_objects(self, frame):
        results = self.model(frame, stream=True)
        detected_objects_info = []

        for r in results:
            boxes = r.boxes

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cls = int(box.cls[0])
                confidence = math.ceil((box.conf[0]*100))/100
                
                obj_info = {
                    "class": self.classNames[cls],
                    "confidence": confidence,
                    "bbox": (x1, y1, x2, y2),
                    "distance": None  # Optional: calculate distance based on bbox and camera parameters
                }
                
                detected_objects_info.append(obj_info)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.putText(frame, f"{self.classNames[cls]}: {confidence*100:.2f}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return detected_objects_info

    def get_near_instructions(self, object_name, min_confidence=50):
        detected_objects_info = self.detect_objects(None) 

        _, frame = self.cap.read()
        frame_height, frame_width = frame.shape[:2]
        current_x, current_y = frame_width // 2, frame_height // 2  # Starting position

        closest_obj_info = None
        closest_distance = float('inf')

        for obj_info in detected_objects_info:
            if obj_info['class'] == object_name and obj_info['confidence'] >= min_confidence:
                target_x, target_y = (obj_info['bbox'][0] + obj_info['bbox'][2]) // 2, (obj_info['bbox'][1] + obj_info['bbox'][3]) // 2  # Center of the bounding box
                dx = target_x - current_x
                dy = target_y - current_y

                distance = dx**2 + dy**2  # Squared distance between the current position and the target

                if distance < closest_distance:
                    closest_distance = distance
                    closest_obj_info = obj_info
                    closest_dx = dx
                    closest_dy = dy

        if closest_obj_info is not None:
            direction_x = "right" if closest_dx > 0 else "left" if closest_dx < 0 else "straight"
            direction_y = "up" if closest_dy > 0 else "down" if closest_dy < 0 else "straight"

            return f"Move {abs(closest_dx)} units {direction_x} and {abs(closest_dy)} units {direction_y} to get near the {object_name}."
        
        return "Object not found."


    def start_detecting(self):
        while True:
            success, img = self.cap.read()
            if not success:
                break
            
            detected_objects_info = self.detect_objects(img)
            
            # Process detected objects here, e.g., display them or use get_near_instructions
            for obj_info in detected_objects_info:
                object_name = obj_info['class']
                instructions = self.get_near_instructions(object_name, min_confidence=80)
                print(instructions)  # Replace this with your voice assistant code
            
            cv2.imshow('Webcam', img)
            if cv2.waitKey(1) == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()