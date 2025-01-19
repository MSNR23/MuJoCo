import mujoco
from mujoco import MjModel, MjData
import numpy as np
import os
from itertools import product

# MuJoCoモデルのロード
model_path = 'g1.xml'  # あなたのMuJoCo XMLファイルパス
model = MjModel.from_xml_path(model_path)
data = MjData(model)

# 定数
L = 1.72  # 身長
ma = 70  # 全体質量
g = 9.80  # 重力加速度

# 上腕リンクの長さ
l1 = 0.186 * L

# 前腕リンクの長さ
l2 = 0.254 * L
# 前腕長さ
l_forearm = 0.575 * l2
# 手長さ
l_hand = 0.425 * l2
# 前腕重さ
m_forearm = 0.016 * ma
# 手重さ
m_hand = 0.006 * ma

# 質点の質量（最後の+は投擲物の重さ）
m1 = 0.028 * ma
m2 = m_forearm + m_hand + 0.14

# 重心までの長さ（m_hand + 投擲物）
lg1 = l1 / 2
lg2 = (m_forearm * (l_forearm / 2) + m_hand * (l_forearm + l_hand / 2) + 0.14 * l2) / (m_forearm + m_hand + 0.14)

# 上腕の慣性モーメント
I1 = m1 * l1**2 / 12
# 前腕の慣性モーメント（平行軸の定理）
I2_forearm = m_forearm * l_forearm**2 / 12 + m_forearm * lg1**2
# 手の慣性モーメント（平行軸の定理）
I2_hand = m_hand * l_hand**2 / 12 + m_hand * (l_forearm + lg2)**2
# 投擲物の慣性モーメント
I2_ball = 0.14 * l2**2
# 前腕リンク全体の慣性モーメント
I2 = I2_forearm + I2_hand + I2_ball

# xmlモデルの長さ、重さ、慣性モーメントをpythonで上書きし、内部的にはpythonのパラメータで計算
# ボディのIDを取得
upper_arm_id = model.body("right_shoulder_pitch_link").id
forearm_id = model.body("light_elbow_link").id

# 重さの上書き
model.body_mass[upper_arm_id] = m1
model.body.mass[forearm_id] = m2

# 慣性モーメントの上書き


# シミュレーション設定
num_episodes = 5000
max_steps_per_episode = 2000
dt = model.opt.timestep  # MuJoCoのタイムステップ

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
    [-20.0, 0.0, 20.0],  # 肩ピッチのトルク範囲
    [-15.0, 0.0, 15.0],  # 肩ロールのトルク範囲
    [-10.0, 0.0, 10.0],  # 肩ヨーのトルク範囲
    [-8.0, 0.0, 8.0],    # 肘ピッチのトルク範囲
]

# 全アクション（3*3*3*3=81通り）
actions = list(product(*torque_ranges))  # 全てのトルク組み合わせを生成

# 離散化関数
def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]

# 状態の離散化関数
def digitize_state(q1, q2, q3, q4, q1_dot, q2_dot, q3_dot, q4_dot):
    digitized = [np.digitize(q1, bins = bins(-np.pi, 50 * np.pi / 180, num_q1_bins)),
                 np.digitize(q2, bins = bins(-np.pi, 0, num_q2_bins)),
                 np.digitize(q3, bins = bins(-60 * np.pi / 180, 80 * np.pi / 180, num_q3_bins)),
                 np.digitize(q4, bins = bins(-60 * np.pi / 180, 90 * np.pi / 180, num_q4_bins)),
                 np.digitize(q1_dot, bins = bins(-10.0, 10.0, num_q1_dot_bins)),
                 np.digitize(q2_dot, bins = bins(-10.0, 10.0, num_q2_dot_bins)),
                 np.digitize(q3_dot, bins = bins(-10.0, 10.0, num_q3_dot_bins)),
                 np.digitize(q4_dot, bins = bins(-10.0, 10.0, num_q4_dot_bins))]
    return tuple(digitized)

# アクション（ε-greedy法）
def get_action(q1_bin, q2_bin, q3_bin, q4_bin, q1_dot_bin, q2_dot_bin, q3_dot_bin, q4_dot_bin, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(num_actions)
    else:
        return np.argmax(Q[q1_bin, q2_bin, q3_bin, q4_bin, q1_dot_bin, q2_dot_bin, q3_dot_bin, q4_dot_bin, :])

# アクション番号に応じたトルク設定
def get_torque(action_idx):
    return actions[action_idx]

# 前腕リンク（手先が接続されているボディ）のID（mujoco内部での計算を出力するためにボディの情報を取得）
forearm_id = model.body("right_elbow_link").id

# 報酬（飛距離とする、手先速度v成分はmujoco側で計算、投射角は前方）
def compute_reward(data, forearm_id, ):
    # mujocoから手先速度を取得（vx,vy,vz）
    v = data.xvelp[forearm_id] # [vx, vy, vz]
    vx, vy, vz = v

    # 速度の合成（ノルムを計算）
    v_syn = np.linalg.norm(v)

    # 投射角（vx, vy, vz成分を用いて示す）
    v_xy = np.sqrt(vx**2 + vy**2)
    angle = np.degrees(np.arctan2(vz, v_xy))

    # 飛距離の計算（0~90度のみ報酬ありで他の角度は叩きつけや後ろ投げのため報酬0、45度で報酬最大（1））
    if 0 <= angle <= 90:
        reward = (v**2 * np.sin(2 * angle)) / g
    else:
        reward = 0

    reward = reward
    reward -= 0.001 * cumulative_energy
    return reward

def reset(data):
    mujoco.mj_resetData(model, data)
    return (data.qpos[0], data.qpos[1], data.qpos[2], data.qpos[3],
            data.qvel[0], data.qvel[1], data.qvel[2], data.qvel[3])

# メインループ
for episode in range(num_episodes):
    state = reset(data)
    cumulative_energy = 0
    epsilon = 0.5 * (0.99 ** (episode + 1))
    max_reward = -float('inf')

    for step in range(max_steps_per_episode):
        q1, q2, q3, q4, q1_dot, q2_dot, q3_dot, q4_dot = state
        q1_bin, q2_bin, q3_bin, q4_bin, q1_dot_bin, q2_dot_bin, q3_dot_bin, q4_dot_bin = digitize_state(q1, q2, q3, q4, q1_dot, q2_dot, q3_dot, q4_dot)

        action = get_action(q1_bin, q2_bin, q3_bin, q4_bin, q1_dot_bin, q2_dot_bin, q3_dot_bin, q4_dot_bin, epsilon)
        torque = get_torque(action)
        data.ctrl[:] = torque  # 各自由度にトルクを適用

        mujoco.mj_step(model, data)  # MuJoCo内部で運動方程式を解く

        # 状態更新
        next_state = (data.qpos[0], data.qpos[1], data.qpos[2], data.qpos[3],
                      data.qvel[0], data.qvel[1], data.qvel[2], data.qvel[3])

        # 報酬計算
        reward = compute_reward(next_state, cumulative_energy)
        Q[q1_bin, q2_bin, q3_bin, q4_bin, q1_dot_bin, q2_dot_bin, q3_dot_bin, q4_dot_bin, action] += alpha * (
            reward + gamma * np.max(Q[q1_bin, q2_bin, q3_bin, q4_bin, q1_dot_bin, q2_dot_bin, q3_dot_bin, q4_dot_bin, :]) -
            Q[q1_bin, q2_bin, q3_bin, q4_bin, q1_dot_bin, q2_dot_bin, q3_dot_bin, q4_dot_bin, action]
        )

        max_reward = max(max_reward, reward)
        state = next_state

    print(f"Episode {episode + 1}/{num_episodes}, Max Reward: {max_reward}")
