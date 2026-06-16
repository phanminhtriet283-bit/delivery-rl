"""
Delivery Route Optimization với Reinforcement Learning
=====================================================
So sánh 3 phương pháp:
1. Greedy (Nearest Neighbor) — baseline
2. Q-Learning (Tabular)
3. DQN (Deep Q-Network)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Cho môi trường không có display (Colab, server)
import os

from env.delivery_env import DeliveryEnv
from agents.greedy import GreedyAgent
from agents.q_learning import QLearningAgent
from agents.dqn import DQNAgent

# ── Config ──────────────────────────────────────────────
N_STOPS = 10
GRID_SIZE = 10
N_EPISODES = 2000
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── 1. Khởi tạo môi trường ──────────────────────────────
env = DeliveryEnv(n_stops=N_STOPS, grid_size=GRID_SIZE)
print("=" * 55)
print("  Delivery Route Optimization — RL Project")
print("=" * 55)
print(f"  Số điểm giao: {N_STOPS} | Grid: {GRID_SIZE}x{GRID_SIZE}")
print(f"  Training episodes: {N_EPISODES}")
print("=" * 55)

# ── 2. Greedy Baseline ───────────────────────────────────
print("\n[1/3] Running Greedy Baseline...")
greedy = GreedyAgent(env)
_, greedy_dist, greedy_route = greedy.run_episode()
print(f"  ✅ Greedy Distance: {greedy_dist:.2f}")

# ── 3. Q-Learning ────────────────────────────────────────
print("\n[2/3] Training Q-Learning...")
ql_agent = QLearningAgent(env, learning_rate=0.1, gamma=0.95,
                          epsilon=1.0, epsilon_decay=0.995)
ql_rewards, ql_distances = ql_agent.train(n_episodes=N_EPISODES, verbose=True)
_, ql_dist, ql_route = ql_agent.run_episode()
print(f"  ✅ Q-Learning Best Distance: {ql_dist:.2f}")

# ── 4. DQN ───────────────────────────────────────────────
print("\n[3/3] Training DQN...")
dqn_agent = DQNAgent(env, lr=1e-3, gamma=0.95,
                     epsilon=1.0, epsilon_decay=0.995, batch_size=64)
dqn_rewards, dqn_distances = dqn_agent.train(n_episodes=N_EPISODES, verbose=True)
_, dqn_dist, dqn_route = dqn_agent.run_episode()
print(f"  ✅ DQN Best Distance: {dqn_dist:.2f}")
dqn_agent.save(os.path.join(RESULTS_DIR, "dqn_model.pth"))

# ── 5. So sánh kết quả ───────────────────────────────────
print("\n" + "=" * 55)
print("  RESULTS COMPARISON")
print("=" * 55)
print(f"  Greedy:     {greedy_dist:.2f}")
print(f"  Q-Learning: {ql_dist:.2f}  ({((greedy_dist - ql_dist)/greedy_dist*100):+.1f}% vs Greedy)")
print(f"  DQN:        {dqn_dist:.2f}  ({((greedy_dist - dqn_dist)/greedy_dist*100):+.1f}% vs Greedy)")
print("=" * 55)

# ── 6. Vẽ Learning Curves ────────────────────────────────
def smooth(data, window=50):
    return np.convolve(data, np.ones(window)/window, mode='valid')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#0f0f23")

for ax in axes:
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333366")

# Reward curve
axes[0].plot(smooth(ql_rewards), color="#4fc3f7", label="Q-Learning", linewidth=1.5)
axes[0].plot(smooth(dqn_rewards), color="#ff6b6b", label="DQN", linewidth=1.5)
axes[0].axhline(y=np.mean(ql_rewards[-200:]), color="#4fc3f7", linestyle="--", alpha=0.5)
axes[0].axhline(y=np.mean(dqn_rewards[-200:]), color="#ff6b6b", linestyle="--", alpha=0.5)
axes[0].set_title("Learning Curve (Reward)", fontsize=13)
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Total Reward")
axes[0].legend(facecolor="#1a1a2e", labelcolor="white")

# Distance curve
axes[1].plot(smooth(ql_distances), color="#4fc3f7", label="Q-Learning", linewidth=1.5)
axes[1].plot(smooth(dqn_distances), color="#ff6b6b", label="DQN", linewidth=1.5)
axes[1].axhline(y=greedy_dist, color="#ffd700", linestyle="--", label=f"Greedy ({greedy_dist:.1f})", linewidth=1.5)
axes[1].set_title("Distance per Episode", fontsize=13)
axes[1].set_xlabel("Episode")
axes[1].set_ylabel("Total Distance")
axes[1].legend(facecolor="#1a1a2e", labelcolor="white")

plt.suptitle("RL Delivery Route Optimization — Training Progress",
             color="white", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "learning_curves.png"),
            dpi=150, bbox_inches="tight", facecolor="#0f0f23")
print(f"\n✅ Saved learning curves → {RESULTS_DIR}/learning_curves.png")

# ── 7. Vẽ Route Comparison ───────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor("#0f0f23")
titles = [f"Greedy (dist={greedy_dist:.1f})",
          f"Q-Learning (dist={ql_dist:.1f})",
          f"DQN (dist={dqn_dist:.1f})"]
routes = [greedy_route, ql_route, dqn_route]

for ax, title, route in zip(axes, titles, routes):
    env.visited = np.ones(N_STOPS)  # mark all visited for coloring
    env.route_history = route
    env._draw_map(ax, route=route)
    ax.set_title(title, color="white", fontsize=12)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333366")
    ax.tick_params(colors="white")

plt.suptitle("Route Comparison", color="white", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "route_comparison.png"),
            dpi=150, bbox_inches="tight", facecolor="#0f0f23")
print(f"✅ Saved route comparison → {RESULTS_DIR}/route_comparison.png")

# ── 8. Tạo GIF animation cho DQN route ──────────────────
print("\n⏳ Generating GIF animation (DQN route)...")
env2 = DeliveryEnv(n_stops=N_STOPS, grid_size=GRID_SIZE)
env2.animate_route(
    dqn_route,
    title=f"DQN Route (dist={dqn_dist:.1f})",
    save_path=os.path.join(RESULTS_DIR, "demo.gif")
)
print(f"✅ Saved animation → {RESULTS_DIR}/demo.gif")
print("\n🎉 Done! Check the results/ folder.")
