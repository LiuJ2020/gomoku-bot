import numpy as np

class Board:
    def __init__(self, size=19):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)  # 0=empty, 1=black, 2=white

    def is_valid_move(self, row, col):
        return 0 <= row < self.size and 0 <= col < self.size and self.board[row, col] == 0

    def make_move(self, row, col, color):
        if not self.is_valid_move(row, col):
            return False
        self.board[row, col] = color
        return True

    def check_win(self, row, col, color):
        directions = [(1,0), (0,1), (1,1), (1,-1)]
        for dr, dc in directions:
            count = 1
            for d in [1, -1]:
                r, c = row, col
                while True:
                    r += dr * d
                    c += dc * d
                    if 0 <= r < self.size and 0 <= c < self.size and self.board[r, c] == color:
                        count += 1
                    else:
                        break
            if count >= 5:
                return True
        return False

    def get_possible_moves(self):
        positions = set()
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] != 0:
                    for dr in range(-1, 2):
                        for dc in range(-1, 2):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < self.size and 0 <= nc < self.size:
                                if self.board[nr, nc] == 0:
                                    positions.add((nr, nc))
        if not positions:
            return [(self.size//2, self.size//2)]
        return list(positions)

    def check_win_anywhere(self, color):
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] == color:
                    if self._count_consecutive(r, c, color, 1, 0) >= 5:
                        return True
                    if self._count_consecutive(r, c, color, 0, 1) >= 5:
                        return True
                    if self._count_consecutive(r, c, color, 1, 1) >= 5:
                        return True
                    if self._count_consecutive(r, c, color, 1, -1) >= 5:
                        return True
        return False

    def _count_consecutive(self, r, c, color, dr, dc):
        count = 0
        for i in range(5):
            nr, nc = r + dr*i, c + dc*i
            if 0 <= nr < self.size and 0 <= nc < self.size and self.board[nr, nc] == color:
                count += 1
            else:
                break
        return count

    def is_full(self):
        return np.all(self.board != 0)

    def copy(self):
        new_board = Board(self.size)
        new_board.board = self.board.copy()
        return new_board
