import time
import math
from pydobotplus import Dobot

device = Dobot(port="/dev/ttyACM0")  # adjust port
device.speed(30, 30)

GRID_SIZE = 86.0/3.0  # mm between cells
START_X, START_Y, START_Z = 308.09, 18.806, -8.086  # adjust these for your workspace
Z_DRAW = -8.086
LIFT_Z = 20

CAMERA_VIEW = (210, 4.777, 5.771)

def move_to_camera_view():
    print(f"[Robot] Moving to camera view position {CAMERA_VIEW}...")
    x, y, z = CAMERA_VIEW
    device.move_to(x, y, z, 0)
    print("[Robot] Camera in position for vision detection.")


def draw_line(x1, y1, x2, y2, z=-8.086, lift_z=20):
    device.move_to(x1, y1, z + lift_z, 0)
    device.move_to(x1, y1, z, 0)
    device.move_to(x2, y2, z, 0)
    device.move_to(x2, y2, z + lift_z, 0)

def draw_grid():
    print("Drawing full Tic Tac Toe grid...")

    CELL = 79.47/3.0
    ROWS, COLS = 3, 3
    TOP_LEFT_X, TOP_LEFT_Y, Z = 308.09, 18.806, -8.086
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
    y = START_Y - (row - 0.5 + 1) * GRID_SIZE
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

def draw_D(segments = 24):
    print(f"Drawing D")
    
    radius = GRID_SIZE / 2 - 3
    cx, cy = START_X - GRID_SIZE * 0.5, START_Y + GRID_SIZE * 0.5

    points = []

    for i in range(segments // 2 + 1):
        theta = 2 * math.pi * i/ segments
        x = cx + math.cos(theta) * radius
        y = cy - math.sin(theta) * radius
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

    z_draw = -8.086
    lift_z = 20

    # Position the H just below the D
    radius = GRID_SIZE / 2 - 3
    cx, cy = START_X - GRID_SIZE * 0.5, START_Y + GRID_SIZE * 0.5

    h_offset = GRID_SIZE * 1.2    # how far below the D
    h_height = 2 * radius         # overall height of H
    h_width = 2 * radius * 0.6    # slightly narrower than D

    hx_center = cx - h_offset
    hy_center = cy

    # Compute key coordinates
    x_top = hx_center + h_height / 2
    x_bottom = hx_center - h_height / 2
    y_left = hy_center + h_width / 2
    y_right = hy_center - h_width / 2
    x_mid = hx_center

    # --- Left vertical line ---
    device.move_to(x_top, y_left, z_draw + lift_z, 0)
    device.move_to(x_top, y_left, z_draw, 0)
    device.move_to(x_bottom, y_left, z_draw, 0)
    device.move_to(x_bottom, y_left, z_draw + lift_z, 0)

    # --- Right vertical line ---
    device.move_to(x_top, y_right, z_draw + lift_z, 0)
    device.move_to(x_top, y_right, z_draw, 0)
    device.move_to(x_bottom, y_right, z_draw, 0)
    device.move_to(x_bottom, y_right, z_draw + lift_z, 0)

    # --- Middle connector ---
    device.move_to(x_mid, y_left, z_draw + lift_z, 0)
    device.move_to(x_mid, y_left, z_draw, 0)
    device.move_to(x_mid, y_right, z_draw, 0)
    device.move_to(x_mid, y_right, z_draw + lift_z, 0)

    print("H drawn ✅")


def draw_R(segments=20):
    """Draw a capital 'R' to the left of D."""
    print("Drawing R...")

    # Reference: D is centered roughly at (cx, cy)
    radius = GRID_SIZE / 2 - 3
    d_cx = START_X - GRID_SIZE * 0.5
    d_cy = START_Y + GRID_SIZE * 0.5

    # Offset R to the left of D
    r_offset = GRID_SIZE * 1.2
    cx = d_cx
    cy = d_cy + r_offset

    # Dimensions of R
    height = 2 * radius
    width = 2 * radius * 0.9

    x_top = cx + height / 2
    x_bottom = cx - height / 2
    y_left = cy + width / 2
    y_right = cy - width / 2
    x_mid = cx  # middle for loop and diagonal

    # --- 1️⃣ Left vertical line ---
    device.move_to(x_top, y_left, Z_DRAW + LIFT_Z, 0)
    device.move_to(x_top, y_left, Z_DRAW, 0)
    device.move_to(x_bottom, y_left, Z_DRAW, 0)
    device.move_to(x_bottom, y_left, Z_DRAW + LIFT_Z, 0)

    # --- 2️⃣ Upper half-circle (the "P" part) ---
    curve_points = []
    for i in range(segments // 2 + 1):
        theta = (math.pi/2 - math.pi * i / (segments // 2))  # 0 → π

        y = cy + math.cos(theta) * (width / 2)
        x = x_mid + height/4 + math.sin(theta) * (height / 4)
        curve_points.append((x, y))

    # Connect from top left to top of curve
    x0, y0 = x_mid, y_left
    device.move_to(x_top, y_left, Z_DRAW + LIFT_Z, 0)
    device.move_to(x_top, y_left, Z_DRAW, 0)

    for (x, y) in curve_points:
        device.move_to(x, y, Z_DRAW, 0)

    # --- 3️⃣ Diagonal leg ---
    # Start at the end of the curve, end at bottom-right
    x_diag_end, y_diag_end = x_bottom, y_right

    device.move_to(x_diag_end, y_diag_end, Z_DRAW, 0)
    device.move_to(x_diag_end, y_diag_end, Z_DRAW + LIFT_Z, 0)

    print("R drawn ✅")


def draw_E():
    """Draw a capital 'E' below the R position."""
    print("Drawing E...")

    z_draw = 8.3
    lift_z = 20

    # Position the H just below the D
    radius = GRID_SIZE / 2 - 3
    cx, cy = START_X - GRID_SIZE * 0.5, START_Y + GRID_SIZE * 0.5

    h_offset = GRID_SIZE * 1.2    # how far below the D and to the left of D
    h_height = 2 * radius         # overall height of H
    h_width = 2 * radius * 0.6    # slightly narrower than D

    hx_center = cx - h_offset
    hy_center = cy + h_offset

    # Compute key coordinates
    x_top = hx_center + h_height / 2
    x_bottom = hx_center - h_height / 2
    y_left = hy_center + h_width / 2
    y_right = hy_center - h_width / 2
    x_mid = hx_center

    # --- Left vertical line ---
    device.move_to(x_top, y_left, z_draw + lift_z, 0)
    device.move_to(x_top, y_left, z_draw, 0)
    device.move_to(x_bottom, y_left, z_draw, 0)
    device.move_to(x_bottom, y_left, z_draw + lift_z, 0)

    # --- Top horizontal line ---
    device.move_to(x_top, y_left, z_draw + lift_z, 0)
    device.move_to(x_top, y_left, z_draw, 0)
    device.move_to(x_top, y_right, z_draw, 0)
    device.move_to(x_top, y_right, z_draw + lift_z, 0)

    # --- Middle horizontal line ---
    device.move_to(x_mid, y_left, z_draw + lift_z, 0)
    device.move_to(x_mid, y_left, z_draw, 0)
    device.move_to(x_mid, y_right, z_draw, 0)
    device.move_to(x_mid, y_right, z_draw + lift_z, 0)

    # --- Middle horizontal line ---
    device.move_to(x_bottom, y_left, z_draw + lift_z, 0)
    device.move_to(x_bottom, y_left, z_draw, 0)
    device.move_to(x_bottom, y_right, z_draw, 0)
    device.move_to(x_bottom, y_right, z_draw + lift_z, 0)

    print("E drawn ✅")


def check_winners(board):
    """
    Checks if there's a winner on the Tic-Tac-Toe board.
    Returns (winner_symbol, winning_cells) or (None, None).
    """
    # Rows and columns
    for i in range(3):
        # Row
        if board[i][0] != "" and board[i][0] == board[i][1] == board[i][2]:
            return board[i][0], [(i, 0), (i, 1), (i, 2)]
        # Column
        if board[0][i] != "" and board[0][i] == board[1][i] == board[2][i]:
            return board[0][i], [(0, i), (1, i), (2, i)]

    # Diagonals
    if board[0][0] != "" and board[0][0] == board[1][1] == board[2][2]:
        return board[0][0], [(0, 0), (1, 1), (2, 2)]
    if board[0][2] != "" and board[0][2] == board[1][1] == board[2][0]:
        return board[0][2], [(0, 2), (1, 1), (2, 0)]

    # No winner yet
    return None, None


def draw_winners(cells, overshoot=5):
    """
    Draws a line across the winning triple of cells.
    """
    import numpy as np
    print(f"[Robot] Drawing win line across {cells}")

    coords = [np.array(move_to_cell_center(r + 1, c + 1)) for (r, c) in cells]
    start, end = coords[0], coords[-1]

    # Extend line a little for a clean visual
    direction = (end - start)
    direction = direction / np.linalg.norm(direction)
    start -= direction * overshoot
    end += direction * overshoot

    x1, y1 = start
    x2, y2 = end

    draw_line(x1, y1, x2, y2, Z_DRAW)
    print("[Robot] Win line drawn ✅")


