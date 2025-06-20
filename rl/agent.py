import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from rl.model import GomokuNet

class RLAgent:
    def __init__(self, board_size=19, lr=1e-3):
        self.board_size = board_size
        self.model = GomokuNet(board_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.memory = []
        self.gamma = 0.99
        self.batch_size = 64
        self.train_steps = 0
        self.loss_history = []

    def select_action(self, board, valid_moves, epsilon=0.1):
        if random.random() < epsilon:
            return random.choice(valid_moves)
        state = self._board_to_tensor(board)
        with torch.no_grad():
            q_values = self.model(state)[0]
        q_values = q_values.detach().cpu().numpy().reshape(self.board_size, self.board_size)
        best_move = max(valid_moves, key=lambda m: q_values[m[0], m[1]])
        return best_move

    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 10000:
            self.memory.pop(0)

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.cat(states)
        next_states = torch.cat(next_states)
        q_values = self.model(states)
        next_q_values = self.model(next_states)
        targets = q_values.clone().detach()
        for i, (a, r, d) in enumerate(zip(actions, rewards, dones)):
            idx = a[0] * self.board_size + a[1]
            if d:
                targets[i, idx] = r
            else:
                targets[i, idx] = r + self.gamma * next_q_values[i].max().item()
        loss = self.criterion(q_values, targets)
        self.loss_history.append(loss.item())
        self.train_steps += 1
        if self.train_steps % 100 == 0:
            print(f"Train step {self.train_steps}, loss: {loss.item():.4f}")
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _board_to_tensor(self, board):
        arr = board.board
        black = (arr == 1).astype(np.float32)
        white = (arr == 2).astype(np.float32)
        tensor = np.stack([black, white])[None, ...]
        return torch.tensor(tensor, dtype=torch.float32)
