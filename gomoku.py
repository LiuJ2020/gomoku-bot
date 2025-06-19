import numpy as np
import pygame
from players import Player
from players import HumanPlayer

BOARD_SIZE = 19
CELL_SIZE = 32
MARGIN = 40
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BG_COLOR = (210, 180, 140)
LINE_COLOR = (0, 0, 0)

class GomokuGame:
    def __init__(self, players):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)  # 0=empty, 1=black, 2=white
        self.players = players
        self.current_player_idx = 0  # 0=black, 1=white
        self.winner = None
        self.running = True

    def switch_player(self):
        self.current_player_idx = 1 - self.current_player_idx

    def is_valid_move(self, row, col):
        return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE and self.board[row, col] == 0

    def make_move(self, row, col):
        if not self.is_valid_move(row, col):
            return False
        color = 1 if self.current_player_idx == 0 else 2
        self.board[row, col] = color
        if self.check_win(row, col, color):
            self.winner = color
            self.running = False
        else:
            self.switch_player()
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
                    if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.board[r, c] == color:
                        count += 1
                    else:
                        break
            if count >= 5:
                return True
        return False

    def get_current_player(self):
        return self.players[self.current_player_idx]

    def run(self):
        pygame.init()
        size = BOARD_SIZE * CELL_SIZE + 2 * MARGIN
        screen = pygame.display.set_mode((size, size))
        pygame.display.set_caption('Gomoku')
        clock = pygame.time.Clock()
        font = pygame.font.SysFont(None, 36)
        
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and isinstance(self.get_current_player(), HumanPlayer):
                    x, y = event.pos
                    row = (y - MARGIN + CELL_SIZE // 2) // CELL_SIZE
                    col = (x - MARGIN + CELL_SIZE // 2) // CELL_SIZE
                    if self.is_valid_move(row, col):
                        self.get_current_player().set_move((row, col))
            player = self.get_current_player()
            move = player.get_move(self.board)
            if move:
                self.make_move(*move)
            self.draw_board(screen, font)
            pygame.display.flip()
            clock.tick(30)
        self.draw_board(screen, font)
        if self.winner:
            winner_text = 'Black wins!' if self.winner == 1 else 'White wins!'
        else:
            winner_text = 'Game Over'
        text = font.render(winner_text, True, (255,0,0))
        screen.blit(text, (MARGIN, 10))
        pygame.display.flip()
        pygame.time.wait(3000)
        pygame.quit()

    def draw_board(self, screen, font):
        screen.fill(BG_COLOR)
        for i in range(BOARD_SIZE):
            pygame.draw.line(screen, LINE_COLOR, (MARGIN, MARGIN + i*CELL_SIZE), (MARGIN + (BOARD_SIZE-1)*CELL_SIZE, MARGIN + i*CELL_SIZE))
            pygame.draw.line(screen, LINE_COLOR, (MARGIN + i*CELL_SIZE, MARGIN), (MARGIN + i*CELL_SIZE, MARGIN + (BOARD_SIZE-1)*CELL_SIZE))
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board[r, c] == 1:
                    pygame.draw.circle(screen, BLACK, (MARGIN + c*CELL_SIZE, MARGIN + r*CELL_SIZE), CELL_SIZE//2 - 2)
                elif self.board[r, c] == 2:
                    pygame.draw.circle(screen, WHITE, (MARGIN + c*CELL_SIZE, MARGIN + r*CELL_SIZE), CELL_SIZE//2 - 2)
        turn_text = 'Black' if self.current_player_idx == 0 else 'White'
        text = font.render(f"Turn: {turn_text}", True, (0,0,0))
        screen.blit(text, (MARGIN, 0))

if __name__ == "__main__":
    black = HumanPlayer('black')
    white = HumanPlayer('white')
    game = GomokuGame([black, white])
    game.run()
