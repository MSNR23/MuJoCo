import mujoco
from mujoco import MjModel, MjData
import numpy as np
import pandas as pd
from itertools import product

# MuJoCoモデルのロード
model_path = '1.90.xml'  # MuJoCo XMLファイルのパス
model = MjModel.from_xml_path(model_path)
data = MjData(model)

# 学習済みQテーブルのロード
q_table_path = 'Q_max_true_reward_episode_9600_step_784.npy'  # 学習済みQテーブルのパス
Q = np.load(q_table_path)

# yaw_ranges.csv の読み込みと辞書変換
yaw_ranges_path = 'yaw_ranges.csv'  # ファイルパス
yaw_ranges_df = pd.read_csv(yaw_ranges_path)
yaw_ranges_dict = {
    (row['Pitch (q1)'], row['Roll (q2)']): (row['Yaw Min'], row['Yaw Max'])
    for _, row in yaw_ranges_df.iterrows()
}

# スプライン補間範囲からヨーの最小値・最大値を取得
def get_yaw_range_from_dict(q1, q2, yaw_ranges_dict):
    key = (round(q1), round(q2))  # ピッチとロールを整数で丸める
    if key in yaw_ranges_dict:
        return yaw_ranges_dict[key]
    else:
        raise ValueError(f"Yaw range not found for Pitch (q1)={key[0]}, Roll (q2)={key[1]}")

# 定数
L = 1.72  # 身長    
dt = model.opt.timestep  # MuJoCoのタイムステップ
g = 9.81

# 各自由度のトルク範囲を定義（肩: 3自由度, 肘: 1自由度）
torque_ranges = [
    [-40.0, -20.0, 0.0, 20.0, 40.0],  # 肩ピッチのトルク範囲
    [-40.0, -20.0, 0.0, 20.0, 40.0],  # 肩ロールのトルク範囲
    [-40.0, -20.0, 0.0, 20.0, 40.0],  # 肩ヨーのトルク範囲
    [-30.0, -15.0, 0.0, 15.0, 30.0],  # 肘ピッチのトルク範囲
]
actions = list(product(*torque_ranges))  # トルク組み合わせを生成

# 離散化関数
def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]

# 状態の離散化関数
def digitize_state(q1, q2, q3, q4, q1_dot, q2_dot, q3_dot, q4_dot):
    digitized = [
        np.digitize(q1, bins=bins(-135 * np.pi / 180, 45 * np.pi / 180, 5)),
        np.digitize(q2, bins=bins(-135 * np.pi / 180, -35 * np.pi / 180, 5)),
        np.digitize(q3, bins=bins(-90 * np.pi / 180, 0, 5)),  # 初期値として -90~0、後でスプラインで制約
        np.digitize(q4, bins=bins(-20 * np.pi / 180, 90 * np.pi / 180, 5)),
        np.digitize(q1_dot, bins=bins(-10.0, 10.0, 2)),
        np.digitize(q2_dot, bins=bins(-10.0, 10.0, 2)),
        np.digitize(q3_dot, bins=bins(-10.0, 10.0, 2)),
        np.digitize(q4_dot, bins=bins(-10.0, 10.0, 2)),
    ]
    return tuple(digitized)

# アクション番号に応じたトルク設定
def get_torque(action_idx, q1, q2):
    """
    アクション番号に対応するトルク値を取得し、肩ヨーのトルクを辞書で制約
    """
    torque = list(actions[action_idx])  # アクションリストからトルクを取得

    # ピッチとロールを辞書に存在する範囲内にクリップ
    q1_clipped = np.clip(np.degrees(q1), -135, 45)
    q2_clipped = np.clip(np.degrees(q2), -135, -35)

    # 辞書からヨー範囲を取得
    yaw_min, yaw_max = get_yaw_range_from_dict(q1_clipped, q2_clipped, yaw_ranges_dict)

    # 肩ヨーのトルクをスプライン範囲に制約
    torque[2] = np.clip(torque[2], yaw_min, yaw_max)
    return torque

# def reset_fixed_state(data):
#     """ 初期姿勢を固定して設定 """
#     mujoco.mj_resetData(model, data)

#     # 固定初期姿勢 (学習時の最適な状態を使用)
#     # fixed_q1 = 0.09577
#     # fixed_q2 = -1.4349
#     # fixed_q3 = -0.85925
#     # fixed_q4 = 1.14371

#     # fixed_q1 = 0* np.pi / 180
#     # fixed_q2 = -115 * np.pi / 180
#     # fixed_q3 = -36 * np.pi / 180
#     # fixed_q4 = 0 * np.pi / 180

#     fixed_q1_dot = 0.0
#     fixed_q2_dot = 0.0
#     fixed_q3_dot = 0.0
#     fixed_q4_dot = 0.0

#     # MuJoCoに初期姿勢を設定
#     data.qpos[:4] = [fixed_q1, fixed_q2, fixed_q3, fixed_q4]
#     data.qvel[:4] = [fixed_q1_dot, fixed_q2_dot, fixed_q3_dot, fixed_q4_dot]

#     return data.qpos[:], data.qvel[:]

# # 初期化関数
def reset(data, Q):
    """
    初期姿勢を設定する関数。
    肩ピッチ(q1)と肩ロール(q2)をランダムに設定し、スプライン補間に基づいて肩ヨー(q3)の範囲を計算。
    """
    mujoco.mj_resetData(model, data)

    # 肩ピッチと肩ロールをランダムに設定し、範囲内にクリップ
    q1_random = np.clip(np.random.uniform(-135 * np.pi / 180, 45 * np.pi / 180), -135 * np.pi / 180, 45 * np.pi / 180)
    q2_random = np.clip(np.random.uniform(-135 * np.pi / 180, -35 * np.pi / 180), -135 * np.pi / 180, -35 * np.pi / 180)

    # スプライン補間で肩ヨー範囲を取得
    yaw_min, yaw_max = get_yaw_range_from_dict(np.degrees(q1_random), np.degrees(q2_random), yaw_ranges_dict)
    q3_random = np.random.uniform(np.radians(yaw_min), np.radians(yaw_max))  # ヨーをスプライン範囲内でランダムに設定

    # 肘ピッチをランダムに設定
    q4_random = np.random.uniform(-20 * np.pi / 180, 90 * np.pi / 180)  # 肘ピッチ: -20° ~ 90°

    # 初期姿勢設定
    initial_pose = {
        "right_shoulder_pitch_joint": q1_random,  # 肩ピッチ
        "right_shoulder_roll_joint": q2_random,  # 肩ロール
        "right_shoulder_yaw_joint": q3_random,  # 肩ヨー（スプライン補間範囲内）
        "right_elbow_joint": q4_random,  # 肘ピッチ
    }

    initial_velocities = {joint: 0.0 for joint in initial_pose.keys()}

    # MuJoCoに初期姿勢を設定
    for joint_name, angle in initial_pose.items():
        joint_id = model.joint(joint_name).qposadr
        data.qpos[joint_id] = angle

    # MuJoCoに初期角速度を設定
    for joint_name, velocity in initial_velocities.items():
        joint_id = model.joint(joint_name).dofadr
        data.qvel[joint_id] = velocity

    return data.qpos[:], data.qvel[:]


# シミュレーション実行
max_steps = 3000
data_list = []

# state = reset_fixed_state(data)  # 初期姿勢を固定

state = reset(data, Q)
for step in range(max_steps):
    time = step * dt
    q1, q2, q3, q4 = data.qpos[:4]
    q1_dot, q2_dot, q3_dot, q4_dot = data.qvel[:4]

    state_bins = digitize_state(q1, q2, q3, q4, q1_dot, q2_dot, q3_dot, q4_dot)
    action_idx = np.argmax(Q[state_bins])  # Qテーブルから最適アクションを取得
    torque = get_torque(action_idx, q1, q2)  # 肩ヨーのスプライン制約を適用
    data.ctrl[:] = torque

    mujoco.mj_step(model, data)

    hand_tip_position = data.site_xpos[model.site('hand_tip').id]
    v_body = data.cvel[model.site_bodyid[model.site('hand_tip').id]][:3]
    omega_body = data.cvel[model.site_bodyid[model.site('hand_tip').id]][3:]
    r = hand_tip_position - data.xpos[model.site_bodyid[model.site('hand_tip').id]]
    hand_tip_velocity = v_body + np.cross(omega_body, r)

    vx, vy, vz = hand_tip_velocity
    v_syn = np.sqrt(vx**2 + vy**2 + vz**2)
    v_xy = np.sqrt(vx**2 + vy**2)
    theta_v = np.degrees(np.arctan2(vz, v_xy))  
    cumulative_energy = np.sum(np.abs(data.ctrl[:model.nu] * data.qvel[:model.nu]) * dt)

    h_release = hand_tip_position[2]
    h_shoulder = 0.818 * L
    h = h_shoulder + h_release
    t = (v_syn * np.sin(theta_v) + np.sqrt(v_syn**2 * (np.sin(theta_v))**2 + 2 * g * h)) / g
    distance = v_syn * np.cos(theta_v) * t if vx >= 0 else 0

    reward = distance - 0.005 * cumulative_energy

    data_list.append([time, q1, q2, q3, q4, q1_dot, q2_dot, q3_dot, q4_dot, h_release, vx, vy, vz, v_syn, theta_v, cumulative_energy, t, distance, reward] + list(torque))

columns = ['time', 'q1', 'q2', 'q3', 'q4', 'q1_dot', 'q2_dot', 'q3_dot', 'q4_dot', 'z_position', 'vx', 'vy', 'vz', 'v_syn', 'theta_v', 'cumulative_energy', 'dis_time', 'distance', 'reward', 'shoulder_pitch_torque', 'shoulder_roll_torque', 'shoulder_yaw_torque', 'elbow_pitch_torque']
df = pd.DataFrame(data_list, columns=columns)
df.to_csv("test2.csv", index=False)
print("Simulation data saved to '0204_0.14_1.72.csv'")
