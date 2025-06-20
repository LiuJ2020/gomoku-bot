import numpy as np
from evaluators import BoardEvaluator
from board import Board
from concurrent.futures import ThreadPoolExecutor, as_completed

class MinimaxBoardEvaluator(BoardEvaluator):
    def __init__(self, heuristic_evaluator, max_depth=2):
        self.max_depth = max_depth
        self.heuristic_evaluator = heuristic_evaluator  # Must be a BoardEvaluator

    def evaluate(self, board, turn):
        # Accepts either a Board instance or a numpy array
        if isinstance(board, Board):
            board_obj = board.copy()
        else:
            board_obj = Board(board.shape[0])
            board_obj.board = board.copy()
        color = 1 if turn == 'black' else 2
        opponent = 2 if color == 1 else 1
        score = self._minimax(board_obj, self.max_depth, -float('inf'), float('inf'), True, color, opponent, turn)
        return score

    def _minimax(self, board_obj, depth, alpha, beta, maximizing, color, opponent, turn):
        if board_obj.check_win_anywhere(color):
            return 100000
        elif board_obj.check_win_anywhere(opponent):
            return -100000
        elif depth == 0 or board_obj.is_full():
            # Use the provided heuristic evaluator
            return self.heuristic_evaluator.evaluate(board_obj, turn)

        moves = board_obj.get_possible_moves()
        if maximizing:
            max_eval = -float('inf')
            def evaluate_move(move):
                new_board = board_obj.copy()
                new_board.make_move(move[0], move[1], color)
                return self._minimax(new_board, depth-1, alpha, beta, False, color, opponent, 'white' if turn == 'black' else 'black')
            with ThreadPoolExecutor() as executor:
                futures = {executor.submit(evaluate_move, move): move for move in moves}
                for future in as_completed(futures):
                    eval = future.result()
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
            return max_eval
        else:
            min_eval = float('inf')
            def evaluate_move(move):
                new_board = board_obj.copy()
                new_board.make_move(move[0], move[1], opponent)
                return self._minimax(new_board, depth-1, alpha, beta, True, color, opponent, 'white' if turn == 'black' else 'black')
            with ThreadPoolExecutor() as executor:
                futures = {executor.submit(evaluate_move, move): move for move in moves}
                for future in as_completed(futures):
                    eval = future.result()
                    min_eval = min(min_eval, eval)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
            return min_eval
