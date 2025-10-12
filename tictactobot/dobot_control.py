import time
from pydobotplus import Dobot

device = Dobot(port="/dev/ttyACM0")  # adjust port
device.speed(30, 30)

GRID_SIZE = 23.33  # mm between cells
START_X, START_Y, START_Z = 200, 0, 0  # adjust these for your workspace

def draw_line(x1, y1, x2, y2, z=-10, lift_z=20):
    device.move_to(x1, y1, z + lift_z, 0)
    device.move_to(x1, y1, z, 0)
    device.grip(True)
    device.move_to(x2, y2, z, 0)
    device.grip(False)
    device.move_to(x1, y1, z + lift_z, 0)

def draw_grid():
    print("Drawing full Tic Tac Toe grid...")

    CELL = 23.33
    ROWS, COLS = 3, 3
    TOP_LEFT_X, TOP_LEFT_Y, Z = 360, 48, 8.3
    BOTTOM_LEFT_X = TOP_LEFT_X - CELL * ROWS
    RIGHT_Y = TOP_LEFT_Y - CELL * COLS

    # --- Outer border (4 sides) ---
    draw_line(TOP_LEFT_X, TOP_LEFT_Y, TOP_LEFT_X, RIGHT_Y, Z)           # left border
    draw_line(TOP_LEFT_X, RIGHT_Y, BOTTOM_LEFT_X, RIGHT_Y, Z)           # top border
    draw_line(BOTTOM_LEFT_X, RIGHT_Y, BOTTOM_LEFT_X, TOP_LEFT_Y, Z)     # right border
    draw_line(BOTTOM_LEFT_X, TOP_LEFT_Y, TOP_LEFT_X, TOP_LEFT_Y, Z)     # bottom border

    # --- Inner vertical lines (2) ---
    for i in range(1, COLS):
        y = TOP_LEFT_Y - i * CELL
        draw_line(TOP_LEFT_X, y, BOTTOM_LEFT_X, y, Z)

    # --- Inner horizontal lines (2) ---
    for j in range(1, ROWS):
        x = TOP_LEFT_X - j * CELL
        draw_line(x, TOP_LEFT_Y, x, RIGHT_Y, Z)

    print("Grid drawn ✅")


def move_to_cell(row, col):
    """Return x, y coordinate for a grid cell."""
    x = START_X + (col - 1) * GRID_SIZE
    y = START_Y - (row - 1) * GRID_SIZE
    return x, y

def draw_x(row, col):
    print(f"Drawing X at ({row}, {col})")
    x, y = move_to_cell(row, col)
    d = GRID_SIZE / 2
    device.move_to(x - d, y - d, -10, 0)
    device.grip(True)
    device.move_to(x + d, y + d, -10, 0)
    device.grip(False)
    device.move_to(x - d, y + d, -10, 0)
    device.grip(True)
    device.move_to(x + d, y - d, -10, 0)
    device.grip(False)

def draw_o(row, col):
    print(f"Drawing O at ({row}, {col})")
    x, y = move_to_cell(row, col)
    radius = GRID_SIZE / 2
    device.draw_circle(x, y, radius)  # if supported
    # or approximate with multiple small lines
