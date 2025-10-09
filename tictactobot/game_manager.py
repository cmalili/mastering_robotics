def check_winner(board):
    lines = []
    # Rows and columns
    lines.extend(board)
    lines.extend([[board[r][c] for r in range(3)] for c in range(3)])
    # Diagonals
    lines.append([board[i][i] for i in range(3)])
    lines.append([board[i][2 - i] for i in range(3)])

    for line in lines:
        if line[0] != "" and line.count(line[0]) == 3:
            return line[0]
    return None

def is_draw(board):
    return all(cell != "" for row in board for cell in row)
