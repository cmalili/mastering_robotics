from serial.tools import list_ports

from pydobot import Dobot


port = list_ports.comports()[0].device
device = Dobot(port=port)

<<<<<<< HEAD
pose = device.get_pose()
=======
pose = device._get_pose()
>>>>>>> 88cd7d2b (Created positioning_robot.py file in lab2)
print(pose)
position = pose.position

device.move_to(position.x + 20, position.y, position.z, position.r, wait=False)
device.move_to(position.x, position.y, position.z, position.r, wait=True)


device.close()