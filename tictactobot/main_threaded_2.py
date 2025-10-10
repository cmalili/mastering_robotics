import cv2
import time
import queue
import threading
import numpy as np
import os
import tempfile
import base64
from inference_sdk import InferenceHTTPClient
from game_manager import check_winner, is_draw
from minimax import compute_best_move
# from dobot_control import draw_x, draw_o   # re-enable later once vision stable

# === Configuration ===
API_URL   = "http://localhost:9001"
WORKSPACE = "chris-hub"
WORKFLOW  = "detect-and-classify-3"
API_KEY   = "1HhSNS3VWex8YfHgeGzJ"

client = InferenceHTTPClient(api_url=API_URL, api_key=API_KEY)
tmp_path = os.path.join(tempfile.gettempdir(), "rf_frame.jpg")

# === Shared Data ===
board_state = [["" for _ in range(3)] for _ in range(3)]
last_frame = None
lock = threading.Lock()
stop_flag = threading.Event()
move_queue = queue.Queue()


# === Helper for drawing boxes ===
def draw_boxes_bgr(img_bgr, preds):
    for p in preds:
        cx, cy, w, h = p["x"], p["y"], p["width"], p["height"]
        x1, y1 = int(cx - w/2), int(cy - h/2)
        x2, y2 = int(cx + w/2), int(cy + h/2)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img_bgr, f'{p["class"]}:{p["confidence"]:.2f}',
                    (x1, max(0,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return img_bgr


# === Vision Thread ===
def vision_loop(cap):
    """Continuously capture frames and run inference."""
    global last_frame, board_state

    print("[Vision] Thread started.")
    while not stop_flag.is_set():
        ok, frame = cap.read()
        if not ok:
            print("[Vision] ⚠️ Frame read failed.")
            time.sleep(0.2)
            continue

        cv2.imwrite(tmp_path, frame)

        try:
            result = client.run_workflow(
                workspace_name=WORKSPACE,
                workflow_id=WORKFLOW,
                images={"image": tmp_path}
            )

            # Parse Roboflow response
            preds = []
            if isinstance(result, list) and len(result) > 0:
                result = result[0]
            if isinstance(result, str):
                import json
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    print("[Vision] ⚠️ Could not parse JSON")
                    continue
            if isinstance(result, dict):
                container = result.get("detection_predictions") or result.get("predictions")
                if container:
                    preds = container.get("predictions", container) if isinstance(container, dict) else container

            # Draw bounding boxes for visualization
            display = draw_boxes_bgr(frame.copy(), preds)
            with lock:
                last_frame = display


            # ---- Parse detections and update board ----
            for det in preds:
                label = det.get("class") or det.get("label")
                x, y = det.get("x"), det.get("y")
                if label not in ["X", "O"] or x is None or y is None:
                    continue

                # Map detection to board cell (assuming grid spans 3x3 evenly)
                row = int((y / frame.shape[0]) * 3)
                col = int((x / frame.shape[1]) * 3)
                row, col = min(max(row, 0), 2), min(max(col, 0), 2)

                # Only add new moves
                if board_state[row][col] == "":
                    board_state[row][col] = label
                    print(f"[Vision] New {label} detected at cell ({row}, {col})")
                    move_queue.put((label, (row, col)))

            

            time.sleep(0.1)

        except Exception as e:
            print("[Vision] Error:", e)
            time.sleep(0.5)

    print("[Vision] Thread stopped.")


# === Game Logic Thread ===
def game_loop():
    global board_state
    previous = [["" for _ in range(3)] for _ in range(3)]
    print("[Game] Thread started.")
    while not stop_flag.is_set():
        with lock:
            board = [r[:] for r in board_state]

        # detect human move (simplified placeholder)
        if board != previous:
            previous = [r[:] for r in board]
            winner = check_winner(board)
            if winner:
                print(f"🏆 Winner: {winner}")
                stop_flag.set()
                return
            if is_draw(board):
                print("🤝 It's a draw!")
                stop_flag.set()
                return

            move = compute_best_move(board)
            if move:
                print("AI move:", move)
                move_queue.put(("O", move))
        time.sleep(1)
    print("[Game] Thread stopped.")


# === (Optional) Robot Thread ===
def robot_loop():
    #from dobot_control import draw_grid, draw_x, draw_o
    from dobot_control_sim import draw_grid, draw_x, draw_o

    print("[Robot] Thread started. Drawing grid...")
    try:
        draw_grid()
        print("[Robot] Grid drawn.")
    except Exception as e:
        print("[Robot] Error while drawing grid:", e)

    # Now wait for AI moves
    while not stop_flag.is_set():
        try:
            symbol, (r, c) = move_queue.get(timeout=1)
            print(f"[Robot] Drawing {symbol} at ({r}, {c})")

            if symbol == "O":
                draw_o(r, c)
            else:
                draw_x(r, c)

            move_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print("[Robot] Error during drawing:", e)

    print("[Robot] Thread stopped.")



# === Main Function ===
def main():
    print("Starting multithreaded Tic Tac Toe (Stable Vision Edition)")
    print("Press 'q' to quit.")

    # initialize camera once
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    # Start threads
    threads = [
        threading.Thread(target=vision_loop, args=(cap,), daemon=True),
        threading.Thread(target=game_loop, daemon=True),
        threading.Thread(target=robot_loop, daemon=True)
    ]
    for t in threads: t.start()

    # --- GUI loop (main thread only) ---
    cv2.namedWindow("Vision", cv2.WINDOW_NORMAL)
    try:
        while not stop_flag.is_set():
            with lock:
                if last_frame is not None:
                    cv2.imshow("Vision", last_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_flag.set()
                break
            time.sleep(0.03)
    except KeyboardInterrupt:
        stop_flag.set()
    finally:
        print("[Main] Stopping...")
        stop_flag.set()
        cap.release()
        cv2.destroyAllWindows()
        for t in threads:
            t.join(timeout=1)
        print("[Main] Clean shutdown complete.")


if __name__ == "__main__":
    main()
