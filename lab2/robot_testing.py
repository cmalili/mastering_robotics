
from pydobot.dobot import MODE_PTP
import pydobot

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

# Preferred way to move

#device.move_to(mode=int(MODE_PTP.MOVJ_XYZ), x=230, y=30, z=20, r=0) # move to position x,y,z
device.move_to(x=230, y=30, z=20, r=0, mode=1)

#device.move_to(mode=int(MODE_PTP.MOVJ_ANGLE), x=20, y=0, z=50, r=0) # move to joint angles (j1=0,j2=0,j3=0,j4=0)
device.move_to(x=20, y=0, z=50, r=0, mode=2)

# Control gripper and suction cup

device.grip(True) # close gripper
device.grip(False) # open gripper

device.suck(True) # Turn on suction cup
device.suck(False) # Turn off suction cup

device.close()