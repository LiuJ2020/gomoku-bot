class Player:
    def __init__(self, color):
        self.color = color  # 'black' or 'white'

    def get_move(self, board):
        # Placeholder: override in subclasses
        raise NotImplementedError
