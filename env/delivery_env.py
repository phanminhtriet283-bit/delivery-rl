import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
import random

class DeliveryEnv(gym.Env):
    """
    Custom Gymnasium Environment for Delivery Route Optimization.
    
    Bài toán: 1 shipper cần giao N đơn hàng.
    Agent học cách chọn thứ tự giao tối ưu để tối thiểu tổng quãng đường.
    
    State:  [vị trí hiện tại (x, y), mask các điểm đã giao]
    Action: chọn điểm giao tiếp theo (0 đến N-1)
    Reward: âm khoảng cách di chuyển (-distance)
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, n_stops=10, grid_size=10, render_mode=None):
        super().__init__()
        self.n_stops = n_stops
        self.grid_size = grid_size
        self.render_mode = render_mode

        # Action space: chọn 1 trong n_stops điểm
        self.action_space = spaces.Discrete(n_stops)

        # Observation space: [x, y hiện tại] + [mask n_stops điểm]
        self.observation_space = spaces.Box(
            low=0,
            high=grid_size,
            shape=(2 + n_stops,),
            dtype=np.float32
        )

        # Sinh điểm giao hàng random cố định (seed để reproduce)
        np.random.seed(42)
        self.stop_locations = np.random.randint(0, grid_size, size=(n_stops, 2)).astype(np.float32)
        self.depot = np.array([grid_size / 2, grid_size / 2], dtype=np.float32)  # Kho xuất phát

        # Lưu lịch sử route để visualize
        self.route_history = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_pos = self.depot.copy()
        self.visited = np.zeros(self.n_stops, dtype=np.float32)  # 0 = chưa giao, 1 = đã giao
        self.total_distance = 0.0
        self.route_history = [self.current_pos.copy()]
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        # Nếu điểm đã giao rồi → phạt nặng
        if self.visited[action] == 1:
            reward = -self.grid_size * 2  # penalty
            obs = self._get_obs()
            terminated = bool(self.visited.sum() == self.n_stops)
            return obs, reward, terminated, False, {}

        # Di chuyển đến điểm action
        next_pos = self.stop_locations[action]
        dist = np.linalg.norm(next_pos - self.current_pos)
        self.total_distance += dist

        # Cập nhật trạng thái
        self.current_pos = next_pos.copy()
        self.visited[action] = 1
        self.route_history.append(self.current_pos.copy())

        reward = -dist  # reward âm = tối thiểu quãng đường

        # Done khi giao hết tất cả điểm
        terminated = bool(self.visited.sum() == self.n_stops)

        # Bonus khi hoàn thành tất cả
        if terminated:
            reward += 10

        obs = self._get_obs()
        return obs, reward, terminated, False, {}

    def _get_obs(self):
        return np.concatenate([self.current_pos, self.visited]).astype(np.float32)

    def get_valid_actions(self):
        """Trả về danh sách các điểm chưa giao."""
        return [i for i in range(self.n_stops) if self.visited[i] == 0]

    def render(self):
        if self.render_mode == "human":
            self._render_frame()

    def _render_frame(self):
        fig, ax = plt.subplots(figsize=(7, 7))
        self._draw_map(ax)
        plt.title(f"Total Distance: {self.total_distance:.2f}")
        plt.tight_layout()
        plt.show()

    def _draw_map(self, ax, route=None):
        ax.set_xlim(-0.5, self.grid_size + 0.5)
        ax.set_ylim(-0.5, self.grid_size + 0.5)
        ax.set_facecolor("#1a1a2e")
        ax.grid(True, color="#333366", linewidth=0.5, alpha=0.5)

        # Vẽ các điểm giao hàng
        for i, (x, y) in enumerate(self.stop_locations):
            color = "#00ff88" if self.visited[i] else "#ff6b6b"
            ax.scatter(x, y, c=color, s=150, zorder=5, edgecolors="white", linewidths=1.5)
            ax.text(x + 0.15, y + 0.15, str(i), color="white", fontsize=8, fontweight="bold")

        # Vẽ kho (depot)
        ax.scatter(*self.depot, c="#ffd700", s=250, marker="*", zorder=6, edgecolors="white", linewidths=1.5)
        ax.text(self.depot[0] + 0.15, self.depot[1] + 0.15, "Depot", color="#ffd700", fontsize=8)

        # Vẽ route nếu có
        route_to_draw = route if route is not None else self.route_history
        if len(route_to_draw) > 1:
            xs = [p[0] for p in route_to_draw]
            ys = [p[1] for p in route_to_draw]
            ax.plot(xs, ys, c="#4fc3f7", linewidth=2, alpha=0.8, zorder=4)

    def animate_route(self, route_positions, title="Delivery Route", save_path=None):
        """
        Tạo animation GIF cho route.
        route_positions: list các (x, y) theo thứ tự di chuyển
        """
        fig, ax = plt.subplots(figsize=(7, 7))
        fig.patch.set_facecolor("#0f0f23")

        def update(frame):
            ax.clear()
            ax.set_facecolor("#1a1a2e")
            ax.grid(True, color="#333366", linewidth=0.5, alpha=0.5)
            ax.set_xlim(-0.5, self.grid_size + 0.5)
            ax.set_ylim(-0.5, self.grid_size + 0.5)
            ax.set_title(f"{title} | Step {frame}/{len(route_positions)-1}",
                        color="white", fontsize=12, pad=10)
            fig.patch.set_facecolor("#0f0f23")

            # Điểm giao hàng
            for i, (x, y) in enumerate(self.stop_locations):
                visited = any(
                    np.allclose([x, y], route_positions[j])
                    for j in range(1, min(frame + 1, len(route_positions)))
                )
                color = "#00ff88" if visited else "#ff6b6b"
                ax.scatter(x, y, c=color, s=150, zorder=5, edgecolors="white", linewidths=1.5)
                ax.text(x + 0.15, y + 0.15, str(i), color="white", fontsize=8)

            # Depot
            ax.scatter(*self.depot, c="#ffd700", s=250, marker="*", zorder=6, edgecolors="white")
            ax.text(self.depot[0] + 0.15, self.depot[1] + 0.15, "Depot", color="#ffd700", fontsize=8)

            # Route đã đi
            if frame > 0:
                xs = [route_positions[i][0] for i in range(frame + 1)]
                ys = [route_positions[i][1] for i in range(frame + 1)]
                ax.plot(xs, ys, c="#4fc3f7", linewidth=2, alpha=0.8, zorder=4)

            # Vị trí hiện tại
            ax.scatter(*route_positions[frame], c="#ffffff", s=200, zorder=7,
                      marker="^", edgecolors="#4fc3f7", linewidths=2)

            # Distance đã đi
            dist = sum(
                np.linalg.norm(np.array(route_positions[i+1]) - np.array(route_positions[i]))
                for i in range(min(frame, len(route_positions)-1))
            )
            ax.text(0.02, 0.98, f"Distance: {dist:.2f}", transform=ax.transAxes,
                   color="#ffd700", fontsize=11, va="top",
                   bbox=dict(boxstyle="round", facecolor="#1a1a2e", alpha=0.8))

        anim = animation.FuncAnimation(fig, update, frames=len(route_positions),
                                       interval=500, repeat=True)
        if save_path:
            anim.save(save_path, writer="pillow", fps=2, dpi=100)
            print(f"✅ Saved animation to {save_path}")
        else:
            plt.show()

        plt.close()
        return anim
