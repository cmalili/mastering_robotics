def compute_best_move(board):
    """Return best next move as (row, col)."""
    # Simple strategy: play first empty spot
    for r in range(3):
        for c in range(3):
            if board[r][c] == "":
                return (r, c)
    return None  # board full
