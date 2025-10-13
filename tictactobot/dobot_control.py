import time
import math
from pydobotplus import Dobot

device = Dobot(port="/dev/ttyACM0")  # adjust port
device.speed(30, 30)

GRID_SIZE = 23.33  # mm between cells
START_X, START_Y, START_Z = 360, 48, 8.3  # adjust these for your workspace
Z_DRAW = 8.3
LIFT_Z = 20

CAMERA_VIEW = (210, -2.26, 61.9)

def move_to_camera_view():
    print(f"[Robot] Moving to camera view position {CAMERA_VIEW}...")
    x, y, z = CAMERA_VIEW
    device.move_to(x, y, z, 0)
    print("[Robot] Camera in position for vision detection.")


def draw_line(x1, y1, x2, y2, z=-10, lift_z=20):
    device.move_to(x1, y1, z + lift_z, 0)
    device.move_to(x1, y1, z, 0)
    device.move_to(x2, y2, z, 0)
    device.move_to(x2, y2, z + lift_z, 0)

def draw_grid():
    print("Drawing full Tic Tac Toe grid...")

    CELL = 23.33333
    ROWS, COLS = 3, 3
    TOP_LEFT_X, TOP_LEFT_Y, Z = 360, 48, 8.3
    BOTTOM_LEFT_X = TOP_LEFT_X - CELL * ROWS
    RIGHT_Y = TOP_LEFT_Y - CELL * COLS

    # --- vertical lines (2) ---
    for i in range(COLS + 1):
        y = TOP_LEFT_Y - i * CELL
        draw_line(TOP_LEFT_X, y, BOTTOM_LEFT_X, y, Z)

    # --- horizontal lines (2) ---
    for j in range(ROWS + 1):
        x = TOP_LEFT_X - j * CELL
        draw_line(x, TOP_LEFT_Y, x, RIGHT_Y, Z)

    print("Grid drawn ✅")


def move_to_cell_center(row, col):
    """Return x, y coordinate for a grid cell."""
    x = START_X - (col - 0.5) * GRID_SIZE
    y = START_Y - (row - 0.5) * GRID_SIZE
    return x, y

def draw_x(row, col):
    print(f"Drawing X at ({row}, {col})")

    cx, cy = move_to_cell_center(row, col)

    half = GRID_SIZE/2 - 3
    draw_line(cx + half, cy + half, cx - half, cy - half)
    draw_line(cx + half, cy - half, cx - half, cy + half)



def draw_o(row, col, segments = 24):
    print(f"Drawing O at ({row}, {col})")
    cx, cy = move_to_cell_center(row, col)
    radius = GRID_SIZE / 2 - 3

    points = []

    for i in range(segments + 1):
        theta = 2 * math.pi / segments
        x = cx + math.cos(theta) * radius
        y = cy + math.sin(theta) * radius
        points.append((x, y))

    x0, y0 = points[0]
    device.move_to(x0, y0, Z_DRAW + LIFT_Z, 0)
    device.move_to(x0, y0, Z_DRAW, 0)

    for (x, y) in points[1:]:
        device.move_to(x, y, Z_DRAW, 0)

    device.move_to(points[-1][0], points[-1][1], Z_DRAW + LIFT_Z, 0)
    print(f"O drawn at ({row}, {col})")


def draw_win():
    print("Drawing a diagonal line across grid")
    draw_line(START_X, START_Y, START_X - 3 * GRID_SIZE, START_Y - 3 * GRID_SIZE)

def draw_draw(segments = 24):
    print(f"Drawing D")
    
    radius = GRID_SIZE / 2 - 3
    cx, cy = START_X - GRID_SIZE * 0.5, START_Y + GRID_SIZE * 0.5

    points = []

    for i in range(segments // 2 + 1):
        theta = 2 * math.pi * i/ segments
        x = cx + math.cos(theta) * radius
        y = cy + math.sin(theta) * radius
        points.append((x, y))

    x0, y0 = points[0]
    device.move_to(x0, y0, Z_DRAW + LIFT_Z, 0)
    device.move_to(x0, y0, Z_DRAW, 0)

    for (x, y) in points[1:]:
        device.move_to(x, y, Z_DRAW, 0)

    device.move_to(x0, y0, Z_DRAW, 0)
    device.move_to(x0, y0, Z_DRAW + LIFT_Z, 0)
    print(f"D drawn)")


def draw_H():
    """Draw a capital 'H' below the D position."""
    print("Drawing H...")

    z_draw = 8.3
    lift_z = 20

    # Position the H just below the D
    radius = GRID_SIZE / 2 - 3
    cx, cy = START_X - GRID_SIZE * 0.5, START_Y + GRID_SIZE * 0.5

    h_offset = GRID_SIZE * 1.2    # how far below the D
    h_height = 2 * radius         # overall height of H
    h_width = 2 * radius * 0.6    # slightly narrower than D

    hx_center = cx
    hy_center = cy - h_offset

    # Compute key coordinates
    x_left = hx_center - h_width / 2
    x_right = hx_center + h_width / 2
    y_top = hy_center + h_height / 2
    y_bottom = hy_center - h_height / 2
    y_mid = hy_center

    # --- Left vertical line ---
    device.move_to(x_left, y_top, z_draw + lift_z, 0)
    device.move_to(x_left, y_top, z_draw, 0)
    device.move_to(x_left, y_bottom, z_draw, 0)
    device.move_to(x_left, y_bottom, z_draw + lift_z, 0)

    # --- Right vertical line ---
    device.move_to(x_right, y_top, z_draw + lift_z, 0)
    device.move_to(x_right, y_top, z_draw, 0)
    device.move_to(x_right, y_bottom, z_draw, 0)
    device.move_to(x_right, y_bottom, z_draw + lift_z, 0)

    # --- Middle connector ---
    device.move_to(x_left, y_mid, z_draw + lift_z, 0)
    device.move_to(x_left, y_mid, z_draw, 0)
    device.move_to(x_right, y_mid, z_draw, 0)
    device.move_to(x_right, y_mid, z_draw + lift_z, 0)

    print("H drawn ✅")

