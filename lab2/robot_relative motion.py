from pydobot.dobot import MODE_PTP
import pydobot

device = pydobot.Dobot(port="/dev/ttyUSB0")
#device = pydobot.Dobot(port="/dev/tty.usbserial-XXXX")

device.home()

device.speed(10,10)  # velocity 10, acceleration 10


# Advanced way to move
#device.move_to(mode=int(MODE_PTP.JUMP_XYZ), x=230, y=30, z=20, r=0) # move to position x,y,z
device.move_to(x=230, y=30, z=20, r=0, mode=1)

#device.move_to(mode=int(MODE_PTP.JUMP_ANGLE), x=200, y=30, z=20, r=0) # move to joint angles (j1=0,j2=0,j3=0,j4=0)
device.move_to(x=200, y=30, z=20, r=0, mode=2)

# Control gripper and suction cup