import numpy as np
import pygame
import pickle
import argparse
import os

def render_sample_game(sample_game, board_size):
    pygame.init()
    CELL_SIZE = 32
    MARGIN = 40
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    BG_COLOR = (210, 180, 140)
    LINE_COLOR = (0, 0, 0)
    size = board_size * CELL_SIZE + 2 * MARGIN
    screen = pygame.display.set_mode((size, size))
    pygame.display.set_caption('Gomoku Sample Game Viewer')
    font = pygame.font.SysFont(None, 36)
    move_idx = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    move_idx = min(move_idx + 1, len(sample_game) - 1)
                elif event.key == pygame.K_LEFT:
                    move_idx = max(move_idx - 1, 0)
        # Draw board up to move_idx
        board = np.zeros((board_size, board_size), dtype=int)
        for i in range(move_idx + 1):
            _, action, _ = sample_game[i]
            color = 1 if i % 2 == 0 else 2
            board[action[0], action[1]] = color
        screen.fill(BG_COLOR)
        for i in range(board_size):
            pygame.draw.line(screen, LINE_COLOR, (MARGIN, MARGIN + i*CELL_SIZE), (MARGIN + (board_size-1)*CELL_SIZE, MARGIN + i*CELL_SIZE))
            pygame.draw.line(screen, LINE_COLOR, (MARGIN + i*CELL_SIZE, MARGIN), (MARGIN + i*CELL_SIZE, MARGIN + (board_size-1)*CELL_SIZE))
        for r in range(board_size):
            for c in range(board_size):
                if board[r, c] == 1:
                    pygame.draw.circle(screen, BLACK, (MARGIN + c*CELL_SIZE, MARGIN + r*CELL_SIZE), CELL_SIZE//2 - 2)
                elif board[r, c] == 2:
                    pygame.draw.circle(screen, WHITE, (MARGIN + c*CELL_SIZE, MARGIN + r*CELL_SIZE), CELL_SIZE//2 - 2)
        move_text = font.render(f"Move: {move_idx+1}/{len(sample_game)}", True, (0,0,0))
        screen.blit(move_text, (MARGIN, 0))
        pygame.display.flip()
    pygame.quit()

def view_sample_games(sample_games_path):
    if sample_games_path and sample_games_path.endswith('.pkl') and os.path.isfile(sample_games_path):
        with open(sample_games_path, 'rb') as f:
            sample_games = pickle.load(f)
        if not sample_games or not isinstance(sample_games, list) or not sample_games[0]:
            print("Invalid or empty sample games file.")
            return
        # Infer board size from the first state
        first_state = sample_games[0][0][0]
        if isinstance(first_state, np.ndarray):
            board_size = first_state.shape[-1]
        else:
            print("Could not infer board size from sample games.")
            return
        print(f"Loaded {len(sample_games)} sample games. Board size inferred as {board_size}.")
        print("Press number keys 1-{} to select a game, or ESC to exit.".format(len(sample_games)))
        while True:
            try:
                inp = input(f"Select game [1-{len(sample_games)}] (or 'q' to quit): ")
                if inp.lower() == 'q':
                    return
                idx = int(inp) - 1
                if 0 <= idx < len(sample_games):
                    render_sample_game(sample_games[idx], board_size)
                else:
                    print("Invalid selection.")
            except Exception as e:
                print(f"Error: {e}")
    else:
        print("Please provide a valid .pkl file containing sample games.")

def main():
    parser = argparse.ArgumentParser(description="View saved sample Gomoku games.")
    parser.add_argument('--timestamp', type=str, required=True, help='Timestamp of the saved sample games to view (e.g. 20240620_153000)')
    args = parser.parse_args()
    base_dir = os.path.join(os.path.dirname(__file__), 'rl', 'saved_models', args.timestamp)
    sample_games_path = os.path.join(base_dir, f'sample_games_{args.timestamp}.pkl')
    if not os.path.isfile(sample_games_path):
        print(f"Sample games file not found: {sample_games_path}")
        return
    view_sample_games(sample_games_path)

if __name__ == "__main__":
    main()

