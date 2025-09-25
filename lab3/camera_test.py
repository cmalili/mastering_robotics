import cv2
import time
import pydobotplus
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



device = pydobotplus.Dobot(port="/dev/ttyACM0")
device.home()
(pose, joint) = device.get_pose()
print(pose)

def detect():

    prev_t = time.time()
    win_name = 'Camera Stream + YOLO'

    start = time.time()
    duration = 0

    labels = []

    while duration < 3:
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

                labels.append(label)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)
                cv2.putText(frame, label, (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                
        now = time.time()
        fps = 1.0 / (now - prev_t)
        prev_t = now

        duration = now - start

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(win_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    return labels


def pick_and_place_with_suction_cup(PICK_UP, DROP, offset=80):
    x1, y1, z1, r1 = PICK_UP
    x2, y2, z2, r2 = DROP

    start = time.time()

    device.move_to(x=x1,y=y1,z=z1+offset,r=r1,mode=1)
    device.move_to(x=x1,y=y1,z=z1,r=r1,mode=1)      # move to block position
    device.suck(True)                               # turn on suction cup
    device.move_to(x=x1,y=y1,z=z1+offset,r=r1,mode=1)      # move vertically up
    device.move_to(x=x2,y=y2,z=z2+offset,r=r2,mode=1)      # move to block destination
    device.move_to(x=x2,y=y2,z=z2,r=r2,mode=1)      # move vertically down to block destination
    device.suck(False)
    time.sleep(2)
    device.move_to(x=x2,y=y2,z=z2+offset,r=r2,mode=1)
    end = time.time()
    return end-start

PICK_UP = [297.90,7.77,-55.34,1.49]

ABOVE_PICK_UP = [297.90,7.77,-55.34 + 30,1.49]

CAMERA_ABOVE = [232.95, 3.14, -9.50, 0.77]

PALLET_A_DROP = [294.29,-129.25,-24.24,-23.71]
PALLET_B_DROP = [295.59,134.94,8.23,24.54]

food = ["banana", "apple", "pizza"]
vehicle = ["bicycle", "airplane", "car"]

for i in range(6):
    # move to pile 
    # detect
    # if object is food move to A, if object is in vehicle move to B
    device.move_to(x=CAMERA_ABOVE[0],y=CAMERA_ABOVE[1],z=CAMERA_ABOVE[2],r=CAMERA_ABOVE[3],mode=1)
    names = detect()
    names = [item.split()[0] for item in names]

    device.move_to(x=ABOVE_PICK_UP[0],y=ABOVE_PICK_UP[1],z=ABOVE_PICK_UP[2],r=ABOVE_PICK_UP[3],mode=1)
    if set(food) & set(names):
        #move to a
        pick_and_place_with_suction_cup(PICK_UP, PALLET_A_DROP)

    elif set(vehicle) & set(names):
        # move to b
        pick_and_place_with_suction_cup(PICK_UP, PALLET_B_DROP)
    else:
        # move home
        device.home()
        
    #device.move_to(x=ABOVE_PICK_UP[0],y=ABOVE_PICK_UP[1],z=ABOVE_PICK_UP[2],r=ABOVE_PICK_UP[3],mode=1)


device.home()
print(names)











cap.release()
cv2.destroyAllWindows()