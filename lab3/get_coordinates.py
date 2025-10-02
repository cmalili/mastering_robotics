import pydobotplus

device = pydobotplus.Dobot(port="/dev/ttyACM0")

device.home()

pose, joint = device.get_pose()

print(pose)

device.speed(10, 10)

# Preferred way to move

#device.move_to(mode=int(MODE_PTP.MOVJ_XYZ), x=230, y=30, z=20, r=0) # move to position x,y,z
#device.move_to(x=230, y=30, z=20, r=0, mode=1)

#device.move_to(mode=int(MODE_PTP.MOVJ_ANGLE), x=20, y=0, z=50, r=0) # move to joint angles (j1=0,j2=0,j3=0,j4=0)
#device.move_to(x=20, y=0, z=50, r=0, mode=2)

# Control gripper and suction cup

#device.grip(True) # close gripper
#device.grip(False) # open gripper

#device.suck(True) # Turn on suction cup
#device.suck(False) # Turn off suction cup

device.close()