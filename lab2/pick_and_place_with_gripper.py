#from pydobot.dobot import MODE_PTP
import pydobotplus
import time

device = pydobotplus.Dobot(port="/dev/ttyACM0")
#device = pydobot.Dobot(port="/dev/tty.usbserial-XXXX")

device.home()

(pose, joint) = device.get_pose()

# position, joint = pose.position, pose.joints

print(pose)

device.speed(10, 10)

def pick_and_place_with_gripper(x1,y1,z1,r1,x2,y2,z2,r2, offset=80):
    start = time.time()

    device.move_to(x=x1,y=y1,z=z1+offset,r=r1,mode=1)
    device.move_to(x=x1,y=y1,z=z1,r=r1,mode=1)      # move to block position
    device.grip(True)                               # turn on suction cup
    device.move_to(x=x1,y=y1,z=z1+offset,r=r1,mode=1)      # move vertically up
    device.move_to(x=x2,y=y2,z=z2+offset,r=r2,mode=1)      # move to block destination
    device.move_to(x=x2,y=y2,z=z2,r=r2,mode=1)      # move vertically down to block destination
    device.grip(False)
    time.sleep(2)
    device.move_to(x=x2,y=y2,z=z2+offset,r=r2,mode=1)
    end = time.time()
    return end-start


x1,y1,z1,r1 = 281.16,-83.78,-12.48,-16.59
x2,y2,z2,r2 = 283.01,-28.23,-12.45,-5.70
x3,y3,z3,r3 = 342.35,-25.73,-14.04,-4.30
x4,y4,z4,r4 = 339.78,-87.17,-15.48,-14.39

x5,y5,z5,r5 = 279.2,14.12,-11.79,2.9
x6,y6,z6,r6 = 283.91,73.03,-12.25,14.43
x7,y7,z7,r7 = 341.70,73.83,-11.71,12.20
x8,y8,z8,r8 = 341.92,12.85,-11.11,2.15

# Corners of pallet number 1
p1_corners = []
p1_corners.append([x1,y1,z1,r1])
p1_corners.append([x2,y2,z2,r2])
p1_corners.append([x3,y3,z3,r3])
p1_corners.append([x4,y4,z4,r4])

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

