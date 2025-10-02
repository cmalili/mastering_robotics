from vision import VisionSystem
from robot import DobotController
import time


FOOD = ["banana", "apple", "pizza"]
VEHICLES = ["car", "bicycle", "airplane"]

# Predefined positions
PICK_UP = [297.90, 7.77, -57.34, 1.49]
ABOVE_PICK_UP = [297.90, 7.77, -25.34, 1.49]
CAMERA_ABOVE = [232.95, 3.14, -9.50, 0.77]
PALLET_A = [294.29,-129.25,-24.24,-23.71]
PALLET_B = [295.59,134.94,8.23,24.54]

def main