
import threading, time, queue, json, cv2
from inference_sdk import InferenceHTTPClient
from dobot_control_sim import draw_grid, draw_x, draw_o, move_to_camera_view
from game_manager import check_winner, is_draw
from minimax import compute_best_move

# === Configuration ===
WORKSPACE = "chris-hub"
WORKFLOW = "detect-and-classify-3"
CAMERA_ID = 0                     # webcam ID
CAMERA_VIEW = (210, -2.26, 61.9)
API_KEY   = "1HhSNS3VWex8YfHgeGzJ"

# === Globals ===
board_state = [["" for _ in range(3)] for _ in range(3)]
latest_detection = None
stop_flag = threading.Event()
lock = threading.Lock()

# === Initialize Roboflow Client ===
client = InferenceHTTPClient(api_url="http://localhost:9001", api_key=API_KEY)

# ===========================================================
# Vision Thread
# ===========================================================
def vision_loop():
    """Continuously captures frames and updates global latest_detection."""
    global latest_detection

    print("[Vision] Thread started.")
    cap = cv2.VideoCapture(CAMERA_ID)
    tmp_path = "frame.jpg"

    while not stop_flag.is_set():
        ok, frame = cap.read()
        if not ok:
            print("[Vision] Frame read failed.")
            time.sleep(0.2)
            continue

        cv2.imwrite(tmp_path, frame)

        try:
            result = client.run_workflow(
                workspace_name=WORKSPACE,
                workflow_id=WORKFLOW,
                images={"image": tmp_path}
            )

            # Parse Roboflow response safely
            if isinstance(result, list) and len(result) > 0:
                result = result[0]
            if isinstance(result, str):
                result = json.loads(result)

            preds = []
            if isinstance(result, dict):
                container = result.get("detection_predictions") or result.get("predictions")
                if container:
                    preds = container.get("predictions", container) if isinstance(container, dict) else container

            # Confidence filtering
            preds = [p for p in preds if p.get("confidence", 0) >= 0.55]

            # Build a new board
            new_board = [["" for _ in range(3)] for _ in range(3)]
            for det in preds:
                label = det.get("class") or det.get("label")
                x, y, w, h = det.get("x"), det.get("y"), det.get("width", 0), det.get("height", 0)
                if not label or x is None or y is None:
                    continue

                # Draw boxes
                start_x = int(x - w / 2)
                start_y = int(y - h / 2)
                end_x = int(x + w / 2)
                end_y = int(y + h / 2)
                color = (0, 255, 0) if label == "X" else (0, 0, 255)
                cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, 2)
                cv2.putText(frame, label, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


                row = min(max(int((y / frame.shape[0]) * 3), 0), 2)
                col = min(max(int((x / frame.shape[1]) * 3), 0), 2)
                new_board[row][col] = label

            with lock:
                latest_detection = new_board

            

        except Exception as e:
            print("[Vision] Error:", e)
            time.sleep(0.5)

        # Display live camera feed
        cv2.imshow("Tic-Tac-Toe Detection", frame)

        # Allow user to quit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_flag.set()
            break

    cap.release()
    print("[Vision] Thread stopped.")

# ===========================================================
# Validation and Helper Functions
# ===========================================================
def get_latest_detection():
    global latest_detection
    with lock:
        return [r[:] for r in latest_detection] if latest_detection else None

def validate_human_move(new_board, human_symbol):
    """Ensure only one valid human move has been added."""
    global board_state
    diff = []
    for r in range(3):
        for c in range(3):
            if board_state[r][c] == "" and new_board[r][c] != "":
                diff.append((r, c, new_board[r][c]))

    if len(diff) != 1:
        print("Invalid: multiple or no new marks detected.")
        return False

    r, c, symbol = diff[0]
    if symbol != human_symbol:
        print(f"Invalid: expected '{human_symbol}' but got '{symbol}'.")
        return False

    board_state[r][c] = human_symbol
    print(f"[Game] Human move accepted at ({r}, {c}).")
    return True

def print_board(board):
    print("\nCurrent board:")
    for row in board:
        print(" | ".join(s if s else " " for s in row))
    print("-" * 9)

# ===========================================================
# Robot/Game Loop (deterministic)
# ===========================================================
def robot_game_loop():
    global board_state

    print("[Robot] Starting game.")
    draw_grid()
    print("[Robot] Grid drawn")

    # Assign default symbols
    ai_symbol, human_symbol = "X", "O"

    # Ask who starts
    choice = ""
    while choice.lower() not in ["robot", "human"]:
        choice = input("Who first: Robot or Human? ").strip().lower()

    if choice == "human":
        ai_symbol, human_symbol = "O", "X"
        move_to_camera_view()   # camera ready for human move
    print(f"[Game] Robot='{ai_symbol}', Human='{human_symbol}'")

    turn = choice

    # --- Main game loop ---
    while not stop_flag.is_set():
        if turn == "robot":
            print("[Robot] Computing best move...")
            move = compute_best_move(board_state, ai_symbol, human_symbol)
            if not move:
                print("It's a draw (no moves left).")
                break

            print(f"[Robot] Drawing {ai_symbol} at {move}")
            if ai_symbol == "X": draw_x(*move)
            else: draw_o(*move)

            board_state[move[0]][move[1]] = ai_symbol
            print_board(board_state)

            if check_winner(board_state):
                print(f"Robot is the Winner: {ai_symbol}")
                break
            if is_draw(board_state):
                print("It's a draw!")
                break

            move_to_camera_view()
            turn = "human"

        elif turn == "human":
            print("[Robot] Waiting for human move...")
            time.sleep(5)  # allow human to draw
            detected = get_latest_detection()
            if detected and validate_human_move(detected, human_symbol):
                print_board(board_state)
                if check_winner(board_state):
                    print(f"Human is the Winner: {human_symbol}")
                    break
                if is_draw(board_state):
                    print("It's a draw!")
                    break
                turn = "robot"
            else:
                print("[Robot] No valid move detected. Retrying...")

        time.sleep(0.5)

    print("[Robot] Game over. Returning to safe position.")
    stop_flag.set()

# ===========================================================
# Main Entry Point
# ===========================================================
if __name__ == "__main__":
    print("Starting deterministic Tic Tac Toe (Robot + Vision)...")

    vision_thread = threading.Thread(target=vision_loop, daemon=True)
    vision_thread.start()

    try:
        robot_game_loop()
    except KeyboardInterrupt:
        print("[Main] Interrupted by user.")
    finally:
        stop_flag.set()
        vision_thread.join()
        print("[Main] Clean shutdown complete.")
