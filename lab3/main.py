from vision import VisionSystem
from robot import DobotController
import time, threading, queue


FOOD = ["banana", "apple", "pizza"]
VEHICLES = ["car", "bicycle", "airplane"]

# Predefined positions
PICK_UP = [297.90, 7.77, -57.34, 1.49]
ABOVE_PICK_UP = [297.90, 7.77, -25.34, 1.49]
CAMERA_ABOVE = [232.95, 3.14, -9.50, 0.77]
PALLET_A = [294.29,-129.25,-24.24,-23.71]
PALLET_B = [295.59,134.94,8.23,24.54]

detection_queue = queue.Queue()


def vision_thread(vision: VisionSystem):
    """continuously make detections and push them to the queue"""
    names = vision.detect(duration=1, show=False)   # run YOLO for 1 second
    if names:
        detection_queue.put(names)
        time.sleep(1)                               # sleep to avoid overloading the cpu


def robot_thread(robot: DobotController):
    """continuously consume detections and act"""
    while True:
        try:
            names = detection_queue.get(timeout=5)
        except queue.Empty:
            print("No detections for 5s, homing and stopping")
            robot.home()
            break

        names = [n.split()[0] for n in names]
        print(f"Robot detectect {names}")

        robot.device.move_to(*ABOVE_PICK_UP, mode=1)

        if set(names) & set(FOOD):
            robot.pick_and_place(PICK_UP, PALLET_A)
            print("Placed food")
        elif set(names) & set(VEHICLES):
            robot.pick_and_place(PICK_UP, PALLET_B)
            print("Placed vehicle")
        else:
            robot.home()
            print("Unknown object")


def main():

    vision = VisionSystem(model_path='yolo8s.pt', conf_thresh=0.5)
    robot = DobotController(port="/dev/ttyACM0")

    try:
        t1 = threading.Thread(target=vision_thread, args=(vision,), daemon=True)
        t2 = threading.Thread(target=robot_thread, args=(robot,), daemon=True)

        t1.start()
        t2.start()

        t2.join()
    
    finally:
        robot.home()
        vision.release()
        print("Task ended")

    if __name__=="__main__":
        main()