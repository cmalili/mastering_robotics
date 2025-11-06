import pydobotplus

device = pydobotplus.Dobot(port="/dev/ttyACM0")

#device.home()

pose, joint = device.get_pose()

print(pose)

device.speed(10, 10)

device.move_to(215.56, -5.8, 53.62, 0)

device.close()