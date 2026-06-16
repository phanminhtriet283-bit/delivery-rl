import numpy as np
import random

class QLearningAgent:
    """
    Tabular Q-Learning Agent.
    Cover CLO3, CLO4: Temporal Difference, Q-learning.
    
    Vì state space lớn (continuous positions + visited mask),
    ta dùng visited mask làm state key (tuple).
    """

    def __init__(self, env, learning_rate=0.1, gamma=0.95,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
        self.env = env
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {}  # dict: state_key -> array [n_stops]

    def _state_key(self, obs):
        """Dùng visited mask làm key cho Q-table."""
        visited = tuple(obs[2:].astype(int))
        return visited

    def _get_q(self, state_key):
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.env.n_stops)
        return self.q_table[state_key]

    def select_action(self, obs, training=True):
        valid_actions = self.env.get_valid_actions()
        if not valid_actions:
            return 0

        # Epsilon-greedy
        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)

        state_key = self._state_key(obs)
        q_values = self._get_q(state_key)

        # Chỉ xét valid actions
        valid_q = {a: q_values[a] for a in valid_actions}
        return max(valid_q, key=valid_q.get)

    def update(self, obs, action, reward, next_obs, done):
        state_key = self._state_key(obs)
        next_key = self._state_key(next_obs)

        q_values = self._get_q(state_key)
        next_q = self._get_q(next_key)

        # Q-learning update: Q(s,a) += lr * [r + gamma * max Q(s') - Q(s,a)]
        target = reward + (0 if done else self.gamma * np.max(next_q))
        q_values[action] += self.lr * (target - q_values[action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, n_episodes=2000, verbose=True):
        rewards_history = []
        distances_history = []

        for ep in range(n_episodes):
            obs, _ = self.env.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.select_action(obs, training=True)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                self.update(obs, action, reward, next_obs, terminated)
                obs = next_obs
                total_reward += reward
                done = terminated or truncated

            self.decay_epsilon()
            rewards_history.append(total_reward)
            distances_history.append(self.env.total_distance)

            if verbose and (ep + 1) % 200 == 0:
                avg_dist = np.mean(distances_history[-200:])
                print(f"Episode {ep+1}/{n_episodes} | Avg Distance (last 200): {avg_dist:.2f} | Epsilon: {self.epsilon:.3f}")

        return rewards_history, distances_history

    def run_episode(self):
        """Chạy 1 episode với greedy policy (không explore) để lấy route."""
        obs, _ = self.env.reset()
        total_reward = 0
        done = False

        while not done:
            action = self.select_action(obs, training=False)
            obs, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += reward
            done = terminated or truncated

        return total_reward, self.env.total_distance, self.env.route_history.copy()
