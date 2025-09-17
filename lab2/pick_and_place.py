from pydobot.dobot import MODE_PTP
import pydobot
import time

device = pydobot.Dobot(port="/dev/ttyUSB0")
#device = pydobot.Dobot(port="/dev/tty.usbserial-XXXX")

device.home()

(pose, joint) = device.get_pose()

print(f"pose : {pose}, joint : {joint}")

[x, y, z, r] = pose

[j1, j2, j3, j4] = joint

# position, joint = pose.position, pose.joints

print(pose)

device.speed(10, 10)



def pick_and_place_with_suction_cup(x1,y1,z1,r1,x2,y2,z2,r2, offset=50):
    start = time.time()
    device.move_to(x=x1,y=y1,z=z1,r=r1,mode=1)      # move to block position
    device.suck(True)                       # turn on suction cup
    device.move_to(x=x1,y=y1,z=z1+offset,r=r1,mode=1)      # move vertically up
    device.move_to(x=x2,y=y2,z=z2-offset,r=r2,mode=1)      # move to block destination
    device.move_to(x=x2,y=y2,z=z2,r=r2,mode=1)      # move vertically down to block destination
    device.suck(False)
    time.sleep(2)
    end = time.time()
    return end-start

def pick_and_place_with_gripper(x1,y1,z1,r1,x2,y2,z2,r2, offset=50):
    start = time.time()
    device.move_to(x=x1,y=y1,z=z1,r=r1,mode=1)      # move to block position
    device.grip(True)                       # turn on suction cup
    device.move_to(x=x1,y=y1,z=z1+offset,r=r1,mode=1)      # move vertically up
    device.move_to(x=x2,y=y2,z=z2-offset,r=r2,mode=1)      # move to block destination
    device.move_to(x=x2,y=y2,z=z2,r=r2,mode=1)      # move vertically down to block destination
    device.grip(False)
    time.sleep(2)
    end = time.time()
    return end-start

