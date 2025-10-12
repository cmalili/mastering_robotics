import time, cv2, json
from inference_sdk import InferenceHTTPClient
from dobot_control_sim import draw_grid, draw_x, draw_o

# --- Roboflow setup ---
WORKSPACE = "chris-hub"
WORKFLOW = "detect-and-classify-3"
client = InferenceHTTPClient(api_url="http://localhost:9001")

# --- Globals ---
board_state = [["" for _ in range(3)] for _ in range(3)]
ai_symbol = "O"
human_symbol = "X"

def print_board(b):
    for row in b:
        print(" ".join([c or "." for c in row]))
    print()

def check_winner(board):
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] != "":
            return board[i][0]
        if board[0][i] == board[1][i] == board[2][i] != "":
            return board[0][i]
    if board[0][0] == board[1][1] == board[2][2] != "":
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != "":
        return board[0][2]
    return None

def is_draw(board):
    return all(cell != "" for row in board for cell in row)

def compute_best_move(board, ai_symbol="O", human_symbol="X"):
    # Simple deterministic move for now
    for r in range(3):
        for c in range(3):
            if board[r][c] == "":
                return (r, c)
    return None

def detect_symbols(frame):
    """Run Roboflow detection and return a 3x3 board matrix."""
    cv2.imwrite("tmp_frame.jpg", frame)
    try:
        result = client.run_workflow(
            workspace_name=WORKSPACE,
            workflow_id=WORKFLOW,
            images={"image": "tmp_frame.jpg"},
        )
    except Exception as e:
        print("[Vision] Error:", e)
        return [["" for _ in range(3)] for _ in range(3)]

    if isinstance(result, list) and len(result) > 0:
        result = result[0]
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except:
            return [["" for _ in range(3)] for _ in range(3)]

    container = result.get("detection_predictions") or result.get("predictions")
    preds = container.get("predictions", container) if isinstance(container, dict) else container

    board = [["" for _ in range(3)] for _ in range(3)]
    if not preds:
        return board

    h, w = frame.shape[:2]
    for det in preds:
        label = det.get("class") or det.get("label")
        conf = det.get("confidence", 0.0)
        x, y = det.get("x"), det.get("y")
        if label not in ["X", "O"] or conf < 0.85:
            continue
        row = int((y / h) * 3)
        col = int((x / w) * 3)
        row, col = min(max(row, 0), 2), min(max(col, 0), 2)
        board[row][col] = label
    return board

def play_deterministic():
    global board_state
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not available.")
        return

    print("[Robot] Drawing grid...")
    draw_grid()
    print("[Robot] Grid drawn.\n")

    choice = ""
    while choice.lower() not in ["robot", "human"]:
        choice = input("Who first: Robot or Human? ").strip().lower()
    turn = choice

    print(f"[Game] {turn.capitalize()} starts.")
    print(f"[Game] Robot='{ai_symbol}', Human='{human_symbol}'")

    while True:
        print_board(board_state)

        if turn == "human":
            input("✏️  Draw your mark and press Enter when ready to capture...")
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed.")
                break

            detected = detect_symbols(frame)
            # Merge detections with existing board
            for r in range(3):
                for c in range(3):
                    if board_state[r][c] == "" and detected[r][c] != "":
                        board_state[r][c] = detected[r][c]

            winner = check_winner(board_state)
            if winner:
                print(f"🏆 Winner: {winner}")
                break
            if is_draw(board_state):
                print("🤝 It's a draw!")
                break

            turn = "robot"

        else:
            move = compute_best_move(board_state, ai_symbol, human_symbol)
            if not move:
                print("No available moves.")
                break
            print(f"[Robot] Playing at {move}")
            r, c = move
            board_state[r][c] = ai_symbol
            draw_o(r, c) if ai_symbol == "O" else draw_x(r, c)

            winner = check_winner(board_state)
            if winner:
                print(f"🏆 Winner: {winner}")
                break
            if is_draw(board_state):
                print("🤝 It's a draw!")
                break

            turn = "human"

    cap.release()
    print("[Game] Finished.")

if __name__ == "__main__":
    play_deterministic()
