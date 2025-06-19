class BoardEvaluator:
    def evaluate(self, board, turn):
        """
        Evaluate the board and return a score indicating how much the player whose turn it is is winning by.
        Positive means advantage for 'turn', negative means advantage for the other player.
        Args:
            board (np.ndarray): 2D numpy array representing the board (0=empty, 1=black, 2=white)
            turn (str): 'black' or 'white'
        Returns:
            int: score (positive = advantage for 'turn', negative = disadvantage)
        """
        raise NotImplementedError