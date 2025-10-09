import time
from pydobotplus import Dobot

device = Dobot(port="/dev/ttyACM0")  # adjust port
device.speed(30, 30)

GRID_SIZE = 60  # mm between cells
START_X, START_Y, START_Z = 200, 0, 0  # adjust these for your workspace

def draw_line(x1, y1, x2, y2, z=-10):
    device.move_to(x1, y1, z)
    device.grip(True)
    device.move_to(x2, y2, z)
    device.grip(False)

def draw_grid():
    print("Drawing Tic Tac Toe grid...")
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
    print("Grid drawn.")

def move_to_cell(row, col):
    """Return x, y coordinate for a grid cell."""
    x = START_X + (col - 1) * GRID_SIZE
    y = START_Y - (row - 1) * GRID_SIZE
    return x, y

def draw_x(row, col):
    print(f"Drawing X at ({row}, {col})")
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

def draw_o(row, col):
    print(f"Drawing O at ({row}, {col})")
    x, y = move_to_cell(row, col)
    radius = GRID_SIZE / 2
    device.draw_circle(x, y, radius)  # if supported
    # or approximate with multiple small lines
