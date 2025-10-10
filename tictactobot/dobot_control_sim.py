import time
import math

# -------------------------------
# Dobot Simulation Module
# -------------------------------
class SimulatedDobot:
    def __init__(self, port=None):
        print(f"[Sim] Dobot initialized on port {port or '/dev/ttyACM0'} (simulation mode)")
        self.speed_value = (0, 0)

    def speed(self, linear, angular):
        self.speed_value = (linear, angular)
        print(f"[Sim] Speed set to linear={linear}, angular={angular}")

    def move_to(self, x, y, z, mode=1):
        print(f"[Sim] Moving to (x={x:.1f}, y={y:.1f}, z={z:.1f}, mode={mode})...")
        time.sleep(0.3)

    def grip(self, state: bool):
        print(f"[Sim] Gripper {'closed' if state else 'opened'}")
        time.sleep(0.2)

    def draw_circle(self, x, y, radius):
        print(f"[Sim] Drawing circle at ({x:.1f}, {y:.1f}) with radius {radius:.1f}")
        steps = 12
        for i in range(steps):
            angle = 2 * math.pi * i / steps
            px = x + radius * math.cos(angle)
            py = y + radius * math.sin(angle)
            print(f"   -> [Sim] Move to ({px:.1f}, {py:.1f})")
            time.sleep(0.05)

# -------------------------------
# Grid Drawing and Symbol Drawing
# -------------------------------

device = SimulatedDobot(port="/dev/ttyACM0")
device.speed(30, 30)

GRID_SIZE = 60
START_X, START_Y, START_Z = 200, 0, 0

def draw_line(x1, y1, x2, y2, z=-10):
    print(f"[Sim] Drawing line from ({x1:.1f}, {y1:.1f}) to ({x2:.1f}, {y2:.1f}) at z={z}")
    device.move_to(x1, y1, z)
    device.grip(True)
    device.move_to(x2, y2, z)
    device.grip(False)
    time.sleep(0.3)

def draw_grid():
    print("[Sim] Drawing Tic Tac Toe grid...")
    # Two vertical lines
    draw_line(START_X + GRID_SIZE, START_Y - GRID_SIZE * 1.5,
              START_X + GRID_SIZE, START_Y + GRID_SIZE * 1.5)
    draw_line(START_X - GRID_SIZE, START_Y - GRID_SIZE * 1.5,
              START_X - GRID_SIZE, START_Y + GRID_SIZE * 1.5)
    # Two horizontal lines
    draw_line(START_X - GRID_SIZE * 1.5, START_Y + GRID_SIZE,
              START_X + GRID_SIZE * 1.5, START_Y + GRID_SIZE)
    draw_line(START_X - GRID_SIZE * 1.5, START_Y - GRID_SIZE,
              START_X + GRID_SIZE * 1.5, START_Y - GRID_SIZE)
    print("[Sim] Grid drawn ✅")

def move_to_cell(row, col):
    """Return x, y coordinate for a grid cell (0-indexed)."""
    x = START_X + (col - 1) * GRID_SIZE
    y = START_Y - (row - 1) * GRID_SIZE
    return x, y

def draw_x(row, col):
    print(f"[Sim] Drawing X at ({row}, {col})")
    x, y = move_to_cell(row, col)
    d = GRID_SIZE / 2
    device.move_to(x - d, y - d, -10)
    device.grip(True)
    device.move_to(x + d, y + d, -10)
    device.grip(False)
    device.move_to(x - d, y + d, -10)
    device.grip(True)
    device.move_to(x + d, y - d, -10)
    device.grip(False)
    print(f"[Sim] X drawn at ({row}, {col})")

def draw_o(row, col):
    print(f"[Sim] Drawing O at ({row}, {col})")
    x, y = move_to_cell(row, col)
    radius = GRID_SIZE / 2
    device.draw_circle(x, y, radius)
    print(f"[Sim] O drawn at ({row}, {col})")
