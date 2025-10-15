import cv2
import numpy as np
from main import draw_grid_overlay

frame = np.zeros((480, 640, 3), dtype=np.uint8)
frame = draw_grid_overlay(frame, origin=(280, 20), grid_size=120, scale=1.0)
cv2.imshow("Grid Test", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
