import cv2
import numpy as np

def detect_grid_size(img_path, debug=False):
    """
    Automatically estimates maze grid size (number of cells per row/column).
    Works by detecting grid lines in the maze image.

    Returns:
        grid_size (int)
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # 1️⃣ Threshold to binary (invert so walls are white)
    _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

    # 2️⃣ Morphological operations to emphasize lines
    kernel = np.ones((3, 3), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # 3️⃣ Detect vertical and horizontal projections
    vertical_sum = np.sum(processed, axis=0)  # sum over rows
    horizontal_sum = np.sum(processed, axis=1)  # sum over columns

    # 4️⃣ Smooth signals and detect peaks (line positions)
    vert_lines = np.where(vertical_sum > 0.5 * np.max(vertical_sum))[0]
    horiz_lines = np.where(horizontal_sum > 0.5 * np.max(horizontal_sum))[0]

    # 5️⃣ Count approximate grid lines
    def count_line_groups(lines, min_gap=10):
        if len(lines) == 0:
            return 0
        count = 1
        prev = lines[0]
        for l in lines[1:]:
            if l - prev > min_gap:
                count += 1
            prev = l
        return count

    n_vert = count_line_groups(vert_lines)
    n_horiz = count_line_groups(horiz_lines)

    # 6️⃣ Grid size = number of cells = (num_lines - 1)
    grid_size = max(n_vert - 1, n_horiz - 1)
    if grid_size < 2:
        grid_size = 4  # fallback default

    if debug:
        print(f"Detected vertical lines: {n_vert}, horizontal lines: {n_horiz}")
        print(f"Estimated grid size: {grid_size}")

        cv2.imshow("Thresholded", thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return grid_size

maze_img = "maze.png"
detected_size = detect_grid_size(maze_img, debug=True)
print("Detected grid size:", detected_size)

#grid, entrance, exit, _ = maze_loader(maze_img, grid_size=detected_size)
