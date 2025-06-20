import numpy as np
import torch
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import pickle
from board import Board
from rl.agent import RLAgent
import argparse
import pygame

def play_single_game(agent, board_size):
    board = Board(board_size)
    state = agent._board_to_tensor(board)
    done = False
    turn = 1  # 1=black, 2=white
    move_count = 0
    win = None
    transitions = []
    def count_patterns(b, color, length):
        count = 0
        for r in range(b.size):
            for c in range(b.size):
                for dr, dc in [(1,0),(0,1),(1,1),(1,-1)]:
                    if all(0 <= r+dr*i < b.size and 0 <= c+dc*i < b.size and b.board[r+dr*i, c+dc*i] == color for i in range(length)):
                        before_r, before_c = r-dr, c-dc
                        after_r, after_c = r+dr*length, c+dc*length
                        before_open = (0 <= before_r < b.size and 0 <= before_c < b.size and b.board[before_r, before_c] == 0)
                        after_open = (0 <= after_r < b.size and 0 <= after_c < b.size and b.board[after_r, after_c] == 0)
                        if before_open or after_open:
                            count += 1
        return count
    while not done:
        valid_moves = board.get_possible_moves()
        action = agent.select_action(board, valid_moves)
        board.make_move(action[0], action[1], turn)
        next_state = agent._board_to_tensor(board)
        reward = 0
        move_count += 1
        my2 = count_patterns(board, turn, 2)
        my3 = count_patterns(board, turn, 3)
        my4 = count_patterns(board, turn, 4)
        opp = 2 if turn == 1 else 1
        opp2 = count_patterns(board, opp, 2)
        opp3 = count_patterns(board, opp, 3)
        opp4 = count_patterns(board, opp, 4)
        reward += 0.01 * my2 + 0.03 * my3 + 0.1 * my4
        reward -= 0.01 * opp2 + 0.03 * opp3 + 0.1 * opp4
        if board.check_win(action[0], action[1], turn):
            reward = 1.0
            done = True
            win = turn
        elif board.is_full():
            reward = 0.0
            done = True
            win = 'draw'
        elif any(board.check_win_anywhere(opp) for _ in [0]):
            reward = -1.0
            done = True
            win = opp
        transitions.append((state, action, reward, next_state, done))
        state = next_state
        turn = 2 if turn == 1 else 1
    return transitions, win, move_count

def train_rl_agent(episodes=10000, board_size=19, parallel_games=4, sample_games_to_save=5):
    agent = RLAgent(board_size)
    win_count = {1: 0, 2: 0, 'draw': 0}
    episode = 0
    sample_games = []
    sample_mod = max(1, episodes // sample_games_to_save)
    while episode < episodes:
        batch_size = min(parallel_games, episodes - episode)
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = [executor.submit(play_single_game, agent, board_size) for _ in range(batch_size)]
            for future in as_completed(futures):
                transitions, win, move_count = future.result()
                for state, action, reward, next_state, done in transitions:
                    agent.store(state, action, reward, next_state, done)
                    agent.train_step()
                win_count[win] += 1
                # Evenly sample games using mod
                if (episode % sample_mod == 0) and (len(sample_games) < sample_games_to_save):
                    sample_games.append([(t[0].numpy(), t[1], t[2]) for t in transitions])
                episode += 1
                if win == 'draw':
                    print(f"Episode {episode}: Draw in {move_count} moves.")
                else:
                    print(f"Episode {episode}: Player {win} wins in {move_count} moves.")
                if episode % 10 == 0:
                    avg_loss = np.mean(agent.loss_history[-100:]) if agent.loss_history else 0
                    print(f"[LOG] Episode {episode} complete. Memory size: {len(agent.memory)} | Avg loss: {avg_loss:.4f} | Wins: {win_count[1]} | {win_count[2]} | Draws: {win_count['draw']}")
    # Save model after training
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    abs_save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'saved_models', timestamp))
    os.makedirs(abs_save_dir, exist_ok=True)
    save_path = os.path.join(abs_save_dir, f'gomoku_rl_{timestamp}.pt')
    torch.save(agent.model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    # Save loss log for visualization
    loss_log_path = os.path.join(abs_save_dir, f'loss_log_{timestamp}.pkl')
    with open(loss_log_path, 'wb') as f:
        pickle.dump(agent.loss_log, f)
    print(f"Loss log saved to {loss_log_path}")
    # Save sample games for visualization
    sample_games_path = os.path.join(abs_save_dir, f'sample_games_{timestamp}.pkl')
    with open(sample_games_path, 'wb') as f:
        pickle.dump(sample_games, f)
    print(f"Sample games saved to {sample_games_path}")
    # Optionally, plot and save the loss curve
    steps, losses = zip(*agent.loss_log) if agent.loss_log else ([],[])
    if steps:
        plt.figure(figsize=(10,5))
        plt.plot(steps, losses)
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        loss_curve_path = os.path.join(abs_save_dir, f'loss_curve_{timestamp}.png')
        plt.savefig(loss_curve_path)
        plt.close()
        print(f"Loss curve saved to {loss_curve_path}")
    return agent

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

def main():
    parser = argparse.ArgumentParser(description="Train RL agent for Gomoku or view sample games.")
    parser.add_argument('--episodes', type=int, default=10000, help='Number of training episodes')
    parser.add_argument('--board_size', type=int, default=19, help='Board size (default: 19)')
    parser.add_argument('--sample_games', type=str, default=None, help='Path to .pkl file containing sample games to view')
    args = parser.parse_args()
    if args.sample_games and args.sample_games.endswith('.pkl') and os.path.isfile(args.sample_games):
        with open(args.sample_games, 'rb') as f:
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
        # Let user pick a game to view
        print("Press number keys 1-{} to select a game, or ESC to exit.".format(len(sample_games)))
        selected = 0
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
        train_rl_agent(episodes=args.episodes, board_size=args.board_size)

if __name__ == "__main__":
    main()
