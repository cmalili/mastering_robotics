import cv2
import time
from ultralytics import YOLO


class VisionSystem:

    def __init__(self, model_path='yolo8n.pt', cam_index=1, conf_thresh=0.5):

        self.cap = cv2.VideoCapture(cam_index)   # Make sure your camera index is correct           
        
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.model = YOLO(model_path)

        #wanted_classes = ["banana", "apple", "pizza", "car", "bicycle", "airplane"]
        #self.allowed_ids = [i for i, name in self.model.names.items() if name in wanted_classes]

        self.conf_thresh = conf_thresh
        self.prev_time = time.time() 

        print(f"Camera initialized: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                                    f"{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}") 
        
    
    def detect(self, duration=1.0, show=True):

        start = time.time()
        labels = []

        while (time.time() - start < duration):

            ret, frame = self.cap.read()
            if not ret :
                print("Frame not received")
                break

            #results = self.model(frame, classes = self.allowed_ids, verbose=False)
            results = self.model(frame, verbose=False)              # specify classes
            r = results[0]

            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf.cpu().numpy())
                    cls = int(box.cls.cpu().numpy())
                    name = r.names[cls]

                    if conf > self.conf_thresh:
                        labels.append(name)
                        
                        cv2.rectangle(frame, (int(x1), int(y1), int(x2), int(y2)), (0, 255,0), 2)
                        cv2.putText(frame, f"{name} {conf:.2f}", (int(x1), int(y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255,0), 2)

            # FPS overlay
            now = time.time()
            fps = 1.0 / (now - self.prev_t)
            self.prev_t = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)

            if show:
                cv2.imshow("YOLO Stream", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        return labels
    

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()