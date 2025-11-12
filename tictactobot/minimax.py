
def compute_best_move(board, ai_symbol="O", human_symbol="X"):
    """
    Return the best move for the AI using the minimax algorithm.
    board: 3x3 list of "X", "O", or "".
    ai_symbol: the symbol for the AI ("O" or "X")
    human_symbol: the opponent's symbol.
    """
    best_score = -float("inf")
    best_move = None

    for r in range(3):
        for c in range(3):
            if board[r][c] == "":
                board[r][c] = ai_symbol
                score = minimax(board, 0, False, ai_symbol, human_symbol)
                board[r][c] = ""
                if score > best_score:
                    best_score = score
                    best_move = (r, c)

    return best_move


def minimax(board, depth, is_maximizing, ai_symbol, human_symbol):
    """Recursive minimax search."""
    winner = check_winner(board)
    if winner == ai_symbol:
        return 10 - depth
    elif winner == human_symbol:
        return depth - 10
    elif is_full(board):
        return 0

    if is_maximizing:
        best_score = -float("inf")
        for r in range(3):
            for c in range(3):
                if board[r][c] == "":
                    board[r][c] = ai_symbol
                    score = minimax(board, depth + 1, False, ai_symbol, human_symbol)
                    board[r][c] = ""
                    best_score = max(best_score, score)
        return best_score
    else:
        best_score = float("inf")
        for r in range(3):
            for c in range(3):
                if board[r][c] == "":
                    board[r][c] = human_symbol
                    score = minimax(board, depth + 1, True, ai_symbol, human_symbol)
                    board[r][c] = ""
                    best_score = min(best_score, score)
        return best_score


def check_winner(board):
    """Return 'X' or 'O' if there's a winner, else None."""
    # Rows and columns
    for i in range(3):
        if board[i][0] != "" and board[i][0] == board[i][1] == board[i][2]:
            return board[i][0]
        if board[0][i] != "" and board[0][i] == board[1][i] == board[2][i]:
            return board[0][i]

    # Diagonals
    if board[0][0] != "" and board[0][0] == board[1][1] == board[2][2]:
        return board[0][0]
    if board[0][2] != "" and board[0][2] == board[1][1] == board[2][0]:
        return board[0][2]

    return None


def is_full(board):
    """Return True if no empty cells remain."""
    return all(board[r][c] != "" for r in range(3) for c in range(3))
