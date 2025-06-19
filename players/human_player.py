from .player import Player

class HumanPlayer(Player):
    def __init__(self, color):
        super().__init__(color)
        self.next_move = None

    def set_move(self, pos):
        self.next_move = pos

    def get_move(self, board):
        move = self.next_move
        self.next_move = None
        return move
