# detects an object below the end effector using the camera
# classify the object using YOLOv5 (e.g food vs vehicle)
# pick up the object using the suction gripper
# place the object in the correct pallet according to its class
# return to home position and repeat cycle

import pydobotplus
import time

device = pydobotplus.Dobot(port="/dev/ttyACM0")
#device = pydobot.Dobot(port="/dev/tty.usbserial-XXXX")

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


# Corners of pallet number 2
p2_corners = []
p2_corners.append([x5,y5,z5,r5])
p2_corners.append([x6,y6,z6,r6])
p2_corners.append([x7,y7,z7,r7])
p2_corners.append([x8,y8,z8,r8])


# moving blocks from first pallet to second pallet
p1_p2_times = [] 
for p1, p2 in zip(p1_corners, p2_corners):
    t = pick_and_place_with_gripper(p1[0],p1[1],p1[2],p1[3],
                                p2[0],p2[1],p2[2],p2[3])
    p1_p2_times.append(t)
    print(f"It takes time : {t}s to move block from corner {p1} to  corner {p2}")


# moving blocks back from second pallet to first pallet
p2_p1_times = []
for p1, p2 in zip(p2_corners, p1_corners):
    t = pick_and_place_with_gripper(p1[0],p1[1],p1[2],p1[3],
                                p2[0],p2[1],p2[2],p2[3])
    p2_p1_times.append(t)
    print(f"It takes time : {t}s to move block from corner {p1} to  corner {p2}")


