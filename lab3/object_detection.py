# detects an object below the end effector using the camera
# classify the object using YOLOv5 (e.g food vs vehicle)
# pick up the object using the suction gripper
# place the object in the correct pallet according to its class
# return to home position and repeat cycle

import pydobotplus
import time
import cv2

from ultralytics import YOLO

device = pydobotplus.Dobot(port="/dev/ttyACM0")

device.home()

(pose, joint) = device.get_pose()

# position, joint = pose.position, pose.joints

print(pose)

HOME = pose
PICK_UP = [0,0,0,0]
PALLET_A_DROP = [0,0,0,0]
PALLET_B_DROP = [0,0,0,0]


device.speed(10, 10)

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

pick_and_place_with_suction_cup(PICK_UP, PALLET_A_DROP)


cap = cv2.VideoCapture(5)


if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Current FPS (reported by driver): ", cap.get(cv2.CAP_PROP_FPS))
print("Current resolution: {}x{}".format(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                                         int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

model = YOLO("yolo5s.pt")