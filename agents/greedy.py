import numpy as np

class GreedyAgent:
    """
    Baseline: Nearest Neighbor Heuristic.
    Luôn chọn điểm giao hàng gần nhất chưa được giao.
    Không học gì cả — dùng để so sánh với RL.
    """

    def __init__(self, env):
        self.env = env

    def select_action(self, obs):
        current_pos = obs[:2]
        visited_mask = obs[2:]

        best_action = None
        best_dist = float("inf")

        for i in range(self.env.n_stops):
            if visited_mask[i] == 0:  # chưa giao
                dist = np.linalg.norm(self.env.stop_locations[i] - current_pos)
                if dist < best_dist:
                    best_dist = dist
                    best_action = i

        return best_action

    def run_episode(self):
        obs, _ = self.env.reset()
        total_reward = 0
        done = False

        while not done:
            action = self.select_action(obs)
            obs, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += reward
            done = terminated or truncated

        return total_reward, self.env.total_distance, self.env.route_history.copy()
