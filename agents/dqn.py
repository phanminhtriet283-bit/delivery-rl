import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    """
    Neural Network xấp xỉ Q-function.
    Cover CLO5, CLO6: Function Approximation bằng Neural Network.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Experience Replay Buffer — giúp DQN học ổn định hơn."""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(obs)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_obs)),
            torch.FloatTensor(dones)
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network Agent.
    Cover CLO5, CLO6: Neural network làm function approximator.
    Dùng Experience Replay + Target Network để training ổn định.
    """

    def __init__(self, env, lr=1e-3, gamma=0.95,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05,
                 batch_size=64, buffer_size=10000, target_update=10):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update = target_update

        input_dim = env.observation_space.shape[0]
        output_dim = env.action_space.n

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Online network + Target network
        self.q_net = QNetwork(input_dim, output_dim).to(self.device)
        self.target_net = QNetwork(input_dim, output_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.loss_fn = nn.MSELoss()
        self.steps = 0

    def select_action(self, obs, training=True):
        valid_actions = self.env.get_valid_actions()
        if not valid_actions:
            return 0

        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)

        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(obs_tensor).squeeze().cpu().numpy()

        # Mask invalid actions
        masked_q = np.full(self.env.n_stops, -np.inf)
        for a in valid_actions:
            masked_q[a] = q_values[a]

        return int(np.argmax(masked_q))

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(self.batch_size)
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_obs = next_obs.to(self.device)
        dones = dones.to(self.device)

        # Current Q values
        q_values = self.q_net(obs).gather(1, actions.unsqueeze(1)).squeeze()

        # Target Q values (dùng target network)
        with torch.no_grad():
            next_q = self.target_net(next_obs).max(1)[0]
            targets = rewards + self.gamma * next_q * (1 - dones)

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, n_episodes=2000, verbose=True):
        rewards_history = []
        distances_history = []
        losses = []

        for ep in range(n_episodes):
            obs, _ = self.env.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.select_action(obs, training=True)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                self.replay_buffer.push(obs, action, reward, next_obs, float(terminated))
                obs = next_obs
                total_reward += reward
                done = terminated or truncated
                self.steps += 1

                loss = self.update()
                if loss:
                    losses.append(loss)

            # Update target network định kỳ
            if ep % self.target_update == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            rewards_history.append(total_reward)
            distances_history.append(self.env.total_distance)

            if verbose and (ep + 1) % 200 == 0:
                avg_dist = np.mean(distances_history[-200:])
                print(f"Episode {ep+1}/{n_episodes} | Avg Distance (last 200): {avg_dist:.2f} | Epsilon: {self.epsilon:.3f}")

        return rewards_history, distances_history

    def run_episode(self):
        obs, _ = self.env.reset()
        total_reward = 0
        done = False

        while not done:
            action = self.select_action(obs, training=False)
            obs, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += reward
            done = terminated or truncated

        return total_reward, self.env.total_distance, self.env.route_history.copy()

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)
        print(f"✅ Model saved to {path}")

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.q_net.state_dict())
        print(f"✅ Model loaded from {path}")
