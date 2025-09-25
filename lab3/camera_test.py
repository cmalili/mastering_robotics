import cv2
import time
from ultralytics import YOLO

cap = cv2.VideoCapture(2)   # Make sure your camera index is correct           
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Current FPS (reported by driver):", cap.get(cv2.CAP_PROP_FPS))
print("Current Resolution: {}x{}".format(
    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
))

model = YOLO('yolov8s.pt')  



prev_t = time.time()
win_name = 'Camera Stream + YOLO'

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting...")
        break

    results = model(frame, verbose=False)  
    r = results[0]

    if r.boxes is not None and len(r.boxes) > 0:

        xyxy = r.boxes.xyxy.cpu().numpy()

        conf = r.boxes.conf.cpu().numpy()
        cls  = r.boxes.cls.cpu().numpy().astype(int)
        names = r.names  

        for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2  
            label = f"{names[k]} {c:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)
            cv2.putText(frame, label, (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            
    now = time.time()
    fps = 1.0 / (now - prev_t)
    prev_t = now
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow(win_name, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()