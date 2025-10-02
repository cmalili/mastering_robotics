# detects an object below the end effector using the camera
# classify the object using YOLOv5 (e.g food vs vehicle)
# pick up the object using the suction gripper
# place the object in the correct pallet according to its class
# return to home position and repeat cycle

import pydobotplus
import time


class DobotController:

    def __init__(self, port="/dev/ttyACM0"):
        self.device = pydobotplus.Dobot(port=port)
        self.device.home()
        print(f"Dobot homed")


    def pick_and_place(self, PICK_UP, DROP, offset=80):
        x1, y1, z1, r1 = PICK_UP
        x2, y2, z2, r2 = DROP

        start = time.time()

        self.device.move_to(x=x1,y=y1,z=z1+offset,r=r1,mode=1)
        self.device.move_to(x=x1,y=y1,z=z1,r=r1,mode=1)      # move to block position
        self.device.suck(True)                               # turn on suction cup
        self.device.move_to(x=x1,y=y1,z=z1+offset,r=r1,mode=1)      # move vertically up
        self.device.move_to(x=x2,y=y2,z=z2+offset,r=r2,mode=1)      # move to block destination
        self.device.move_to(x=x2,y=y2,z=z2,r=r2,mode=1)      # move vertically down to block destination
        self.device.suck(False)
        time.sleep(2)
        self.device.move_to(x=x2,y=y2,z=z2+offset,r=r2,mode=1)
        end = time.time()
        return end-start

    def home(self):
        self.device.home()


cap = cv2.VideoCapture(5)


if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Current FPS (reported by driver): ", cap.get(cv2.CAP_PROP_FPS))
print("Current resolution: {}x{}".format(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                                         int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

model = YOLO("yolo5s.pt")

prev_t = time.time()

win_name = 'Camera Stream + YOLO'

def detect_image():

    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting...")
        #break

    results = model(frame, verbose=False)  
    r = results[0]
    names = r.names

    return names

