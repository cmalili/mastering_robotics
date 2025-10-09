from vision_module import detect_board_state, to_grid
from minimax import compute_best_move
from game_manager import check_winner, is_draw
import time
import cv2

def print_board(board):
    for row in board:
        print(row)
    print()

def main():
    print("Starting Tic Tac Toe AI...")
    print("Press Ctrl+C to stop.")
    while True:
        # 1. Detect board
        preds = detect_board_state()
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break 
        cv2.destroyAllWindows()

        board = to_grid(preds)
        print_board(board)

        # 2. Check game status
        winner = check_winner(board)
        if winner:
            print(f"🏆 Winner: {winner}")
            break
        if is_draw(board):
            print("🤝 It's a draw!")
            break

        # 3. Compute and announce move
        move = compute_best_move(board)
        if move:
            print(f"AI plays at: {move}")
        else:
            print("No valid move left.")
            break

        # 4. Wait for your next move
        print("Your turn! Make your move and press Enter.")
        input()
        time.sleep(1)

if __name__ == "__main__":
    main()
