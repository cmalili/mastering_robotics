import cv2

# --- open camera (0 = default webcam) ---
cap = cv2.VideoCapture(2)

if not cap.isOpened():
    raise IOError("Cannot open camera")

frame_idx = 0
target_idx = 1000
saved = False

print("[INFO] Press 'q' to quit early.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Unable to read frame from camera.")
        break

    frame_idx += 1

    # Display the live feed
    cv2.imshow("Live Feed", frame)

    # Save the 100th frame
    if frame_idx == target_idx:
        cv2.imwrite("data/images/frame_100.png", frame)
        print("✅ Saved frame_100.png")
        saved = True

    # Press 'q' to exit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Quitting before frame 100.")
        break

    # Optionally stop right after saving
    if saved:
        print("[INFO] Stopping after saving frame 100.")
        break

# --- cleanup ---
cap.release()
cv2.destroyAllWindows()
