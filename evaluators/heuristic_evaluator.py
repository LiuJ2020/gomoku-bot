from evaluators import BoardEvaluator
from board import Board
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class HeuristicBoardEvaluator(BoardEvaluator):
    def evaluate(self, board, turn):
        # Accepts either a Board instance or a numpy array
        if isinstance(board, Board):
            board_obj = board
        else:
            board_obj = Board(board.shape[0])
            board_obj.board = board.copy()
        color = 1 if turn == 'black' else 2
        opponent = 2 if color == 1 else 1
        return self._score(board_obj, color, opponent)

    def _score(self, board_obj, color, opponent):
        # Heuristic: count open 2s, 3s, 4s, 5s for color minus for opponent
        def count_patterns(b, col, length):
            count = 0
            positions = [(r, c, dr, dc) for r in range(b.size) for c in range(b.size) for dr, dc in [(1,0),(0,1),(1,1),(1,-1)]]
            def check_pattern(args):
                r, c, dr, dc = args
                return 1 if self._pattern_at(b, r, c, col, length, dr, dc) else 0
            with ThreadPoolExecutor() as executor:
                results = executor.map(check_pattern, positions)
            return sum(results)
        score = 0
        # Weights: open 2s, 3s, 4s, 5s
        weights = {2: 10, 3: 100, 4: 1000, 5: 100000}
        for l, w in weights.items():
            score += w * (count_patterns(board_obj, color, l) - count_patterns(board_obj, opponent, l))
        return score

    def _pattern_at(self, board_obj, r, c, color, length, dr, dc):
        for i in range(length):
            nr, nc = r + dr*i, c + dc*i
            if not (0 <= nr < board_obj.size and 0 <= nc < board_obj.size) or board_obj.board[nr, nc] != color:
                return False
        # Check for open ends
        before_r, before_c = r - dr, c - dc
        after_r, after_c = r + dr*length, c + dc*length
        before_open = (0 <= before_r < board_obj.size and 0 <= before_c < board_obj.size and board_obj.board[before_r, before_c] == 0)
        after_open = (0 <= after_r < board_obj.size and 0 <= after_c < board_obj.size and board_obj.board[after_r, after_c] == 0)
        return before_open or after_open
