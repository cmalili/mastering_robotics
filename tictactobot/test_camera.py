import cv2

cap = cv2.VideoCapture("/dev/video2")  # or the correct index
if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame grab failed")
        break

    cv2.imshow("Robot Camera Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
