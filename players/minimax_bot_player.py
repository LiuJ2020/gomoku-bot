from players.player import Player
from evaluators import *
from board import Board
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

class MinimaxBotPlayer(Player):
    def __init__(self, color, evaluator, max_depth=2):
        super().__init__(color)
        self.evaluator = MinimaxBoardEvaluator(evaluator, max_depth)
        self.max_depth = max_depth

    def get_move(self, board):
        # Accepts either a Board instance or a numpy array
        if isinstance(board, Board):
            board_obj = board.copy()
        else:
            board_obj = Board(board.shape[0])
            board_obj.board = board.copy()
        color = 1 if self.color == 'black' else 2
        opponent = 2 if color == 1 else 1
        best_score = -float('inf')
        best_move = None
        moves = board_obj.get_possible_moves()

        def evaluate_move(move):
            new_board = board_obj.copy()
            new_board.make_move(move[0], move[1], color)
            score = self.evaluator._minimax(new_board, self.max_depth-1, -float('inf'), float('inf'), False, color, opponent, 'white' if self.color == 'black' else 'black')
            return (score, move)

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(evaluate_move, move) for move in moves]
            for future in as_completed(futures):
                score, move = future.result()
                if score > best_score:
                    best_score = score
                    best_move = move
        return best_move
