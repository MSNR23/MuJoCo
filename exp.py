import mujoco
from mujoco import MjSim, MjViewer, load_model_from_path
import numpy as np
import os
import csv
from itertools import product

# MuJoCoモデルのロード
model_path = 'g1.xml'
model = load_model_from_path(model_path)
sim = MjSim(model)
viewer = MjViewer(sim)

# 定数
L = 1.90  # 身長
ma = 70  # 全体質量
g = 9.80  # 重力加速度

# リンクの長さ
l1 = 0.186 * L
l2 = 0.254 * L

# シミュレーション設定
num_episodes = 5000
max_steps_per_episode = 2000
dt = sim.model.opt.timestep  # MuJoCoのタイムステップ

# 保存ディレクトリ
save_dir = "mujoco_results"
os.makedirs(save_dir, exist_ok=True)

# Q学習のパラメータ
alpha = 0.1
gamma = 0.9

# Qテーブルのパラメータ
num_q1_bins = 5
num_q2_bins = 5
num_q3_bins = 5
num_q4_bins = 5
num_q1_dot_bins = 4
num_q2_dot_bins = 4
num_q3_dot_bins = 4
num_q4_dot_bins = 4
num_actions = 81

# Qテーブル
Q = np.zeros((num_q1_bins, num_q2_bins, num_q3_bins, num_q4_bins, num_q1_dot_bins, num_q2_dot_bins, num_q3_dot_bins, num_q4_dot_bins, num_actions))

# 各自由度のトルク範囲を定義（肩: 3自由度, 肘: 1自由度）
torque_ranges = [
    [-20.0, 0.0, 20.0],  # 肩の自由度1のトルク範囲
    [-15.0, 0.0, 15.0],  # 肩の自由度2のトルク範囲
    [-10.0, 0.0, 10.0],  # 肩の自由度3のトルク範囲
    [-8.0, 0.0, 8.0],    # 肘の自由度のトルク範囲
]

# アクション空間を生成（全81通り）
actions = list(product(*torque_ranges))  # 全てのトルク組み合わせを生成

# 離散化関数
def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]

# 状態の離散化関数
def digitize_state(q1, q2, q3, q4,q1_dot, q2_dot, q3_dot, q4_dot):
    digitized = [np.digitize(q1, bins = bins(-np.pi, 50 * np.pi / 180, num_q1_bins)),
                 np.digitize(q2, bins = bins(-np.pi, 0, num_q2_bins)),
                 np.digitize(q3, bins = bins(-60 * np.pi / 180, 80 * np.pi / 180, num_q3_bins)),
                 np.digitize(q4, bins = bins(-60 * np.pi / 180, 90 * np.pi / 180, num_q4_bins)),
                 np.digitize(q1_dot, bins = bins(-10.0, 10.0, num_q1_dot_bins)),
                 np.digitize(q2_dot, bins = bins(-10.0, 10.0, num_q2_dot_bins)),
                 np.digitize(q3_dot, bins = bins(-10.0, 10.0, num_q3_dot_bins)),
                 np.digitize(q4_dot, bins = bins(-10.0, 10.0, num_q4_dot_bins))]

    return tuple(digitized)

# ε-greedy法
def get_action(q1_bin, q2_bin, q3_bin, q4_bin, q1_dot_bin, q2_dot_bin, q3_dot_bin, q4_dot_bin, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(num_actions)
    else:
        return np.argmax(Q[q1_bin, q2_bin, q3_bin, q4_bin, q1_dot_bin, q2_dot_bin, q3_dot_bin, q4_dot_bin, :])

# アクション番号に応じたトルク設定
def get_torque(action_idx):
    """
    アクション番号に対応するトルクを取得
    :param action_idx: アクション番号 (0 ~ 80)
    :return: 各自由度に適用するトルクのリスト
    """
    return actions[action_idx]

def compute_reward(state, cumulative_energy):
    # MuJoCoから取得する状態で報酬を計算
    q1, q2, q3, q4, q1_dot, q2_dot, q3_dot, q4_dot = state
    v_x2 = -l1 * np.sin(q1) * q1_dot - l2 * np.sin(q1 + q2) * (q1_dot + q2_dot)
    v_y2 = l1 * np.cos(q1) * q1_dot + l2 * np.cos(q1 + q2) * (q1_dot + q2_dot)
    v2 = np.sqrt(v_x2**2 + v_y2**2)

    qv = np.arctan2(v_x2, v_y2)
    qv_deg = np.degrees(qv) % 360

    if 0 <= qv_deg <= 45:
        angle_reward = qv_deg / 45
    elif 45 < qv_deg <= 90:
        angle_reward = (90 - qv_deg) / 45
    else:
        angle_reward = 0

    reward = angle_reward * v2
    reward -= 0.001 * cumulative_energy
    return reward

def reset(sim):
    sim.reset()
    sim.set_state(sim.get_state())
    return sim.data.qpos[0], sim.data.qpos[1], sim.data.qpos[2], sim.data.qpos[3], sim.data.qvel[0], sim.data.qvel[1], sim.data.qvel[2], sim.data.qvel[3]

# メインループ
for episode in range(num_episodes):
    state = reset(sim)
    cumulative_energy = 0
    epsilon = 0.5 * (0.99 ** (episode + 1))
    max_reward = -float('inf')

    for step in range(max_steps_per_episode):
        q1, q2, q3, q4, q1_dot, q2_dot, q3_dot, q4_dot = state
        q1_bin, q2_bin, q3_bin, q4_bin, q1_dot_bin, q2_dot_bin, q3_dot_bin, q4_dot_bin = digitize_state(q1, q2, q3, q4,q1_dot, q2_dot, q3_dot, q4_dot)

        action = get_action(q1_bin, q2_bin, q3_bin, q4_bin, q1_dot_bin, q2_dot_bin, q3_dot_bin, q4_dot_bin, epsilon)
        torque = get_torque(action)  # アクション番号に応じたトルクを取得
        sim.data.ctrl[:] = torque  # 各自由度にトルクを適用

        sim.step() # MuJoCo内部で運動方程式を解く→次の状態を計算
        
        # 状態更新
        next_state = (
            sim.data.qpos[0], sim.data.qpos[1], sim.data.qpos[2], sim.data.qpos[3],
            sim.data.qvel[0], sim.data.qvel[1], sim.data.qvel[2], sim.data.qvel[3]
        )

        # 報酬計算
        reward = compute_reward(next_state, cumulative_energy)
        Q[q1_bin, q2_bin, q1_dot_bin, q2_dot_bin, action] += alpha * (
            reward + gamma * np.max(Q[q1_bin, q2_bin, q1_dot_bin, q2_dot_bin, :]) -
            Q[q1_bin, q2_bin, q1_dot_bin, q2_dot_bin, action]
        )

        max_reward = max(max_reward, reward)
        state = next_state

    print(f"Episode {episode + 1}/{num_episodes}, Max Reward: {max_reward}")
