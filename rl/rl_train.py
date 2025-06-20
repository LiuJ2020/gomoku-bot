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
    save_path = os.path.join('saved_models', f'gomoku_rl_{timestamp}.pt')
    torch.save(agent.model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    # Save loss log for visualization
    loss_log_path = os.path.join('saved_models', f'loss_log_{timestamp}.pkl')
    with open(loss_log_path, 'wb') as f:
        pickle.dump(agent.loss_log, f)
    print(f"Loss log saved to {loss_log_path}")
    # Save sample games for visualization
    sample_games_path = os.path.join('saved_models', f'sample_games_{timestamp}.pkl')
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
        plt.savefig(os.path.join('saved_models', f'loss_curve_{timestamp}.png'))
        plt.close()
        print(f"Loss curve saved to saved_models/loss_curve_{timestamp}.png")
    return agent

def main():
    parser = argparse.ArgumentParser(description="Train RL agent for Gomoku.")
    parser.add_argument('--episodes', type=int, default=10000, help='Number of training episodes')
    parser.add_argument('--board_size', type=int, default=19, help='Board size (default: 19)')
    args = parser.parse_args()
    train_rl_agent(episodes=args.episodes, board_size=args.board_size)

if __name__ == "__main__":
    main()
