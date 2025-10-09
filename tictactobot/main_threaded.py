import time
import cv2
import queue
import threading
from vision_module import detect_board_state, to_grid
from minimax import compute_best_move
from game_manager import check_winner, is_draw
from dobot_control import draw_x, draw_o

# === Shared resources ===
board_state = [["" for _ in range(3)] for _ in range(3)]
lock = threading.Lock()
move_queue = queue.Queue()
stop_flag = threading.Event()


# === Vision Thread ===
def vision_loop():
    """Continuously read frames and update the shared board state."""
    global board_state
    while not stop_flag.is_set():
        try:
            display, preds = detect_board_state()
            print("Frame processed.")
            new_board = to_grid(preds)

            # show the current frame
            #cv2.imshow("TicTacToe Detector", display)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    stop_flag.set()
            #    break

            with lock:
                board_state = new_board
                print("board updated")

            # small delay to limit API calls
            time.sleep(0.5)
        except Exception as e:
            print("Vision thread error:", e)
            time.sleep(1)


# === Game Logic Thread ===
def game_loop():
    """Watch for human moves, compute AI responses, and enqueue robot actions."""
    global board_state
    last_board = [["" for _ in range(3)] for _ in range(3)]

    while not stop_flag.is_set():
        with lock:
            board = [row[:] for row in board_state]

        # detect change (human played)
        if board != last_board:
            print("\nDetected board update:")
            for row in board:
                print(row)
            last_board = [r[:] for r in board]

            winner = check_winner(board)
            if winner:
                print(f"🏆 Winner detected: {winner}")
                stop_flag.set()
                return
            if is_draw(board):
                print("🤝 It's a draw!")
                stop_flag.set()
                return

            move = compute_best_move(board)
            if move:
                print("AI move:", move)
                move_queue.put(("O", move))  # Enqueue robot action
            else:
                print("No valid moves remaining.")
                stop_flag.set()
                return

        time.sleep(1)


# === Robot Thread ===
def robot_loop():
    """Consume queued robot moves and execute them sequentially."""
    while not stop_flag.is_set():
        try:
            symbol, (r, c) = move_queue.get(timeout=1)
            print(f"Robot drawing {symbol} at cell ({r}, {c})")
            if symbol == "O":
                draw_o(r, c)
            else:
                draw_x(r, c)
            move_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print("Robot thread error:", e)


# === Main Entry Point ===
def main():
    print("Starting multithreaded Tic Tac Toe AI system.")
    print("Press Ctrl+C to stop.\n")

    threads = [
        threading.Thread(target=vision_loop, daemon=True),
        #threading.Thread(target=game_loop, daemon=True),
        #threading.Thread(target=robot_loop, daemon=True),
    ]

    for t in threads:
        t.start()

    try:
        while not stop_flag.is_set():
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received, stopping...")
        stop_flag.set()

    print("Waiting for threads to finish...")
    for t in threads:
        t.join(timeout=2)

    print("System shut down cleanly.")


if __name__ == "__main__":
    main()
