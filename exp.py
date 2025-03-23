import mujoco
from mujoco import MjModel, MjData
import numpy as np
import os
from itertools import product
import pandas as pd
from scipy.interpolate import RectBivariateSpline

# MuJoCoモデルのロード
model_path = 'g1.xml'  # あなたのMuJoCo XMLファイルパス
model = MjModel.from_xml_path(model_path)
data = MjData(model)

# 定数
L = 1.72  # 身長
ma = 70  # 全体質量
# g = 9.80  # 重力加速度

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

# 保存ディレクトリ
save_dir = "1.72_0.14"
# info_dir = os.path.join(save_dir, "episode_info")  # 各エピソードの情報保存用
qtable_dir = os.path.join(save_dir, "qtables")  # Qテーブル保存用
reward_dir = os.path.join(save_dir, "reward_progress")  # 報酬遷移保存用
pos_dir = os.path.join(save_dir, "ini_pos")  # 報酬遷移保存用

# ディレクトリ作成
# os.makedirs(info_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)
os.makedirs(qtable_dir, exist_ok=True)
os.makedirs(reward_dir, exist_ok=True)
os.makedirs(pos_dir, exist_ok=True)

# xmlモデルの長さ、重さ、慣性モーメントをpythonで上書きし、内部的にはpythonのパラメータで計算
# 上腕
upper_arm_body_id = model.body('right_shoulder_pitch_link').id
model.body_mass[upper_arm_body_id] = m1
model.body_inertia[upper_arm_body_id] = [I1, I1, I1]
model.body_ipos[upper_arm_body_id] = [0, 0, -lg1]  # 重心位置を設定

# 前腕
forearm_body_id = model.body('right_elbow_link').id
model.body_mass[forearm_body_id] = m2
model.body_inertia[forearm_body_id] = [I2, I2, I2]
model.body_ipos[forearm_body_id] = [0, 0, -lg2]  # 重心位置を設定

# 手先の情報を取得
site_id = model.site('hand_tip').id
body_id = model.site_bodyid[site_id]

# 手先位置の取得
hand_tip_position = data.site_xpos[site_id]

# 手先速度の取得
v_body = data.cvel[body_id][:3] # 手モデル中心の線速度
omega_body = data.cvel[body_id][3:] # 手モデル中心の角速度
r = hand_tip_position - data.xpos[body_id] # 手先位置 - ボディ中心位置
hand_tip_velosity = v_body + np.cross(omega_body, r)

# # 初期姿勢の設定（例: 肩と肘の角度を設定）
# initial_pose = {
#     "right_shoulder_pitch_joint": 0.0,  # 初期角度 (度)
#     "right_shoulder_roll_joint": 0.0,
#     "right_shoulder_yaw_joint": 0.0,
#     "right_elbow_joint": 90.0,
# }

# initial_velocities = {
#     "right_shoulder_pitch_joint": 0.0,  # 初期角速度 (rad/s)
#     "right_shoulder_roll_joint": 0.0,
#     "right_shoulder_yaw_joint": 0.0,
#     "right_elbow_joint": 0.0,
# }

# 事前計算の範囲（1°単位でサンプリング）
q1_range = np.arange(-135, 46, 1)  # -135° から 45° まで 1°単位
q2_range = np.arange(-135, -34, 1) # -135° から -35° まで 1°単位

# スプライン補間の準備
def load_and_create_spline(file_path):
    df = pd.read_excel(file_path, header=None)
    pitch_angles = df.iloc[1:, 0].to_numpy()  # A列: 肩ピッチ
    roll_angles = df.iloc[0, 1:].to_numpy()  # 1行: 肩ロール
    yaw_values = df.iloc[1:, 1:].to_numpy()  # 値部分: ヨー角度
    return RectBivariateSpline(pitch_angles, roll_angles, yaw_values)

# yaw_max.xlsx と yaw_min.xlsx からスプライン補間を作成
yaw_max_spline = load_and_create_spline('yaw_max.xlsx')
yaw_min_spline = load_and_create_spline('yaw_min.xlsx')

def get_yaw_range_from_spline(pitch, roll):
    pitch = np.clip(pitch, q1_range.min(), q1_range.max())
    roll = np.clip(roll, q2_range.min(), q2_range.max())
    yaw_min = yaw_min_spline(pitch, roll)[0][0]
    yaw_max = yaw_max_spline(pitch, roll)[0][0]
    return yaw_min, yaw_max

# 事前計算: q1, q2 の組み合わせごとに可動範囲を保存
def precompute_yaw_ranges():
    yaw_ranges = {}
    for q1 in q1_range:  # q1: -135 ~ 45 (1°刻み)
        for q2 in q2_range:  # q2: -135 ~ -35 (1°刻み)
            try:
                yaw_max = yaw_max_spline(q1, q2)[0][0]  # スプライン補間
                yaw_min = yaw_min_spline(q1, q2)[0][0]
                yaw_ranges[(q1, q2)] = (yaw_min, yaw_max)
            except Exception as e:
                print(f"Failed to compute yaw_range for q1={q1}, q2={q2}: {e}")
                yaw_ranges[(q1, q2)] = (0.0, 0.0)  # デフォルト値
    return yaw_ranges


# 事前計算結果の保存
yaw_ranges = precompute_yaw_ranges()

# yaw_min, yaw_max の範囲をCSVに保存
def save_yaw_ranges_to_csv(yaw_ranges, save_path):
    rows = []
    for (q1, q2), (yaw_min, yaw_max) in yaw_ranges.items():
        rows.append([q1, q2, yaw_min, yaw_max])
    df = pd.DataFrame(rows, columns=["Pitch (q1)", "Roll (q2)", "Yaw Min", "Yaw Max"])
    df.to_csv(save_path, index=False)
    print(f"Yaw ranges saved to {save_path}")

# 事前計算結果をCSVに保存
yaw_csv_path = os.path.join(save_dir, "yaw_ranges.csv")
save_yaw_ranges_to_csv(yaw_ranges, yaw_csv_path)

# 事前計算した可動範囲を取得する関数
def get_yaw_range_from_precomputed(q1, q2):
    q1_rounded = int(np.round(q1))  # q1, q2 はすでに degrees 渡される
    q2_rounded = int(np.round(q2))
    q1_rounded = np.clip(q1_rounded, -135, 45)  # 範囲内にクリップ
    q2_rounded = np.clip(q2_rounded, -135, -35)
    
    if (q1_rounded, q2_rounded) not in yaw_ranges:
        raise ValueError(f"Key ({q1_rounded}, {q2_rounded}) not found in precomputed yaw_ranges.")
    
    return yaw_ranges[(q1_rounded, q2_rounded)]


# シミュレーション設定
num_episodes = 10000
max_steps_per_episode = 3000
dt = model.opt.timestep  # MuJoCoのタイムステップ(dt = 0.001)

# # 保存ディレクトリ
# save_dir = "0129_0.14c"
# # info_dir = os.path.join(save_dir, "episode_info")  # 各エピソードの情報保存用
# qtable_dir = os.path.join(save_dir, "qtables")  # Qテーブル保存用
# reward_dir = os.path.join(save_dir, "reward_progress")  # 報酬遷移保存用

# ディレクトリ作成
# os.makedirs(info_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)
os.makedirs(qtable_dir, exist_ok=True)
os.makedirs(reward_dir, exist_ok=True)

# 報酬遷移データの初期化
reward_progress = []

# Q学習のパラメータ
alpha = 0.1
gamma = 0.9

# Qテーブルのパラメータ(3^8*1225=8037225)
num_q1_bins = 5
num_q2_bins = 5
num_q3_bins = 5
num_q4_bins = 5
num_q1_dot_bins = 2
num_q2_dot_bins = 2
num_q3_dot_bins = 2
num_q4_dot_bins = 2
num_actions = 625

# Qテーブル
Q = np.zeros((num_q1_bins, num_q2_bins, num_q3_bins, num_q4_bins, num_q1_dot_bins, num_q2_dot_bins, num_q3_dot_bins, num_q4_dot_bins, num_actions))

# 各自由度のトルク範囲を定義（肩: 3自由度, 肘: 1自由度）
torque_ranges = [
    [-40.0, -20.0, 0.0, 20.0, 40.0],  # 肩ピッチのトルク範囲
    [-40.0, -20.0, 0.0, 20.0, 40.0],  # 肩ロールのトルク範囲
    [-40.0, -20.0, 0.0, 20.0, 40.0],  # 肩ヨーのトルク範囲
    [-30.0, -15.0, 0.0, 15.0, 30.0],    # 肘ピッチのトルク範囲
]

# 全アクション（3*3*3*3=81通り）
actions = [list(action) for action in product(*torque_ranges)] # 全てのトルク組み合わせを生成

# 離散化関数
def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]

# 状態の離散化関数
def digitize_state(q1, q2, q3, q4, q1_dot, q2_dot, q3_dot, q4_dot):
    yaw_min, yaw_max = get_yaw_range_from_precomputed(np.degrees(q1), np.degrees(q2))

    digitized = [
        np.digitize(q1, bins=bins(-135 * np.pi / 180, 45 * np.pi / 180, num_q1_bins)),
        np.digitize(q2, bins=bins(-135 * np.pi / 180, -35 * np.pi / 180, num_q2_bins)),
        np.digitize(q3, bins=bins(np.radians(yaw_min), np.radians(yaw_max), num_q3_bins)),
        np.digitize(q4, bins=bins(-20 * np.pi / 180, 90 * np.pi / 180, num_q4_bins)),
        np.digitize(q1_dot, bins=bins(-10.0, 10.0, num_q1_dot_bins)),
        np.digitize(q2_dot, bins=bins(-10.0, 10.0, num_q2_dot_bins)),
        np.digitize(q3_dot, bins=bins(-10.0, 10.0, num_q3_dot_bins)),
        np.digitize(q4_dot, bins=bins(-10.0, 10.0, num_q4_dot_bins)),
    ]
    return tuple(digitized)


# アクション（ε-greedy法）
def get_action(q1_bin, q2_bin, q3_bin, q4_bin, q1_dot_bin, q2_dot_bin, q3_dot_bin, q4_dot_bin, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(num_actions)
    else:
        return np.argmax(Q[q1_bin, q2_bin, q3_bin, q4_bin, q1_dot_bin, q2_dot_bin, q3_dot_bin, q4_dot_bin, :])

def get_torque(action_idx, q1, q2):
    torque = actions[action_idx][:]  # リストのコピー
    yaw_min, yaw_max = get_yaw_range_from_precomputed(np.degrees(q1), np.degrees(q2))
    torque[2] = np.clip(torque[2], yaw_min, yaw_max)  # 肩ヨーのトルクを制約
    return torque


# 報酬（飛距離とする、手先速度v成分はmujoco側で計算、投射角は前方）
def compute_reward(data, site_id, cumulative_energy, g=9.81):

    # 手先速度を取得（vx, vy, vz）
    vx, vy, vz = hand_tip_velosity

    # 速度成分の合成
    v_syn = np.sqrt(vx**2 + vy**2 + vz**2)
    v_xy = np.sqrt(vx**2 + vy**2)

    # 投射角計算
    theta_v = np.arctan2(vz, v_xy)

    # 手先の高さ（z座標）を取得
    h_release = hand_tip_position[2]

    # リリース高さの計算
    h_shoulder = 0.818 * L
    h = h_shoulder + h_release
    
    # 投射時間
    t = (v_syn * np.sin(theta_v) + np.sqrt(v_syn**2 * (np.sin(theta_v))**2 + 2 * g * h)) / g

    # 投射方向が-90~90度（右方向（前方向））のみで報酬を与える
    if vx >= 0:
        distance = v_syn * np.cos(theta_v) * t
    else:
        distance = 0

    # 報酬 = 飛距離 - 累積消費エネルギー
    reward = distance - 0.005 * cumulative_energy

    return reward

def get_random_yaw_within_spline(q1, q2):
    """
    ピッチ(q1)とロール(q2)の値からスプライン補間でヨー範囲を取得し、その範囲内でランダムな値を返す
    """
    yaw_min, yaw_max = get_yaw_range_from_spline(q1, q2)
    yaw_random = np.random.uniform(yaw_min, yaw_max)
    return yaw_random

# シミュレーションループ
def reset(data):
    # ピッチとロールをランダムに設定
    q1_random = np.random.uniform(-135, 45)  # 肩ピッチ: -50° ～ 30°
    q2_random = np.random.uniform(-135, -35) # 肩ロール: -120° ～ 0°

    # ヨー軸をスプライン補間の範囲内でランダムに設定
    q3_random = get_random_yaw_within_spline(q1_random, q2_random)

    # 肘の角度をランダムに設定
    q4_random = np.random.uniform(-20, 90)  # 肘: -60° ～ 120°

    # 初期姿勢
    initial_pose = {
        "right_shoulder_pitch_joint": q1_random,  # ピッチ
        "right_shoulder_roll_joint": q2_random,  # ロール
        "right_shoulder_yaw_joint": q3_random,   # ヨー
        "right_elbow_joint": q4_random           # 肘
    }

    # 初期角速度はすべて 0 に固定
    initial_velocities = {
        "right_shoulder_pitch_joint": 0.0,
        "right_shoulder_roll_joint": 0.0,
        "right_shoulder_yaw_joint": 0.0,
        "right_elbow_joint": 0.0
    }

    # 初期姿勢を設定
    for joint_name, angle in initial_pose.items():
        joint_id = model.joint(joint_name).qposadr
        data.qpos[joint_id] = np.radians(angle)  # 度からラジアンに変換して設定

    # 初期角速度を設定
    for joint_name, velocity in initial_velocities.items():
        joint_id = model.joint(joint_name).dofadr
        data.qvel[joint_id] = velocity

    return (q1_random, q2_random, q3_random, q4_random,
            data.qpos[0], data.qpos[1], data.qpos[2], data.qpos[3],
            data.qvel[0], data.qvel[1], data.qvel[2], data.qvel[3])

# 報酬の再計算
def compute_true_reward(Q, state_bins):
    """
    学習終了時のQテーブルを用いて報酬を再計算
    """
    q1_bin, q2_bin, q3_bin, q4_bin, q1_dot_bin, q2_dot_bin, q3_dot_bin, q4_dot_bin = state_bins
    true_reward = np.max(Q[q1_bin, q2_bin, q3_bin, q4_bin, q1_dot_bin, q2_dot_bin, q3_dot_bin, q4_dot_bin, :]) 
    return true_reward

# メインループ
max_true_reward = -float('inf')  # 1000エピソード内での最大のTrue Reward
best_q_table = None  # 最大True Rewardに対応するQテーブル

# エピソードごとの初期姿勢を保存するリスト
episode_initial_poses = []

for episode in range(num_episodes):
    q1_init, q2_init, q3_init, q4_init, *state = reset(data)

    # 初期姿勢を記録
    episode_initial_poses.append([episode + 1, q1_init, q2_init, q3_init, q4_init])
    
    # 各エピソードごとに初期姿勢を ini_pos フォルダに保存
    pose_df = pd.DataFrame([[episode + 1, q1_init, q2_init, q3_init, q4_init]],
                           columns=['Episode', 'q1_init', 'q2_init', 'q3_init', 'q4_init'])
    pose_df.to_csv(os.path.join(pos_dir, f"episode_init_pose_{episode + 1}.csv"), index=False)

    cumulative_energy = 0
    epsilon = 0.5 * (0.99 ** (episode + 1))
    max_reward = -float('inf')
    release_step = -1
    episode_data = []
    episode_rewards = []
    max_q_table = None


    for step in range(max_steps_per_episode):
        q1, q2, q3, q4, q1_dot, q2_dot, q3_dot, q4_dot = state
        q1_bin, q2_bin, q3_bin, q4_bin, q1_dot_bin, q2_dot_bin, q3_dot_bin, q4_dot_bin = digitize_state(q1, q2, q3, q4, q1_dot, q2_dot, q3_dot, q4_dot)

        # 事前計算済みの yaw_min, yaw_max を取得
        yaw_min, yaw_max = get_yaw_range_from_precomputed(q1, q2)

        # アクションの選択と対応するトルクの適用
        action = get_action(q1_bin, q2_bin, q3_bin, q4_bin, q1_dot_bin, q2_dot_bin, q3_dot_bin, q4_dot_bin, epsilon)
        torque = get_torque(action, q1, q2)
        data.ctrl[:] = torque  # 各自由度にトルクを適用

        # MuJoCo内部で運動方程式を解き、1ステップ実行
        mujoco.mj_step(model, data) 

        # 状態更新
        next_state = (data.qpos[0], data.qpos[1], data.qpos[2], data.qpos[3],
                      data.qvel[0], data.qvel[1], data.qvel[2], data.qvel[3])

        # 累積消費エネルギー (|トルク × 各リンクの角速度| × dt)
        shoulder_p_energy_consumed = abs(data.ctrl[0] * data.qvel[0]) * dt
        shoulder_r_energy_consumed = abs(data.ctrl[1] * data.qvel[1]) * dt
        shoulder_y_energy_consumed = abs(data.ctrl[2] * data.qvel[2]) * dt
        elbow_p_energy_consumed = abs(data.ctrl[3] * data.qvel[3]) * dt
    
        upper_arm_energy_consumed = shoulder_p_energy_consumed + shoulder_r_energy_consumed + shoulder_y_energy_consumed
        forearm_energy_consumed = elbow_p_energy_consumed

        cumulative_energy += upper_arm_energy_consumed + forearm_energy_consumed

        # 手先位置、速度を取得
        hand_tip_position = data.site_xpos[site_id]
        v_body = data.cvel[body_id][:3] # 手モデル中心の線速度
        omega_body = data.cvel[body_id][3:] # 手モデル中心の角速度
        r = hand_tip_position - data.xpos[body_id] # 手先位置 - ボディ中心位置
        hand_tip_velosity = v_body + np.cross(omega_body, r)
        vx, vy, vz = hand_tip_velosity

        # 投射角度
        v_syn = np.sqrt(vx**2 + vy**2 + vx**2)
        v_xy = np.sqrt(vx**2 + vy**2)
        theta_v = np.degrees(np.arctan2(vz, v_xy))

        # リリース高さ
        h_release = hand_tip_position[2]
        h_shoulder = 0.818 * L
        h = h_shoulder + h_release

        # 飛距離計算
        if vx >= 0:  # 飛距離は正方向のみに計算
            t_flight = (v_syn * np.sin(theta_v) + np.sqrt(v_syn**2 * (np.sin(theta_v))**2 + 2 * 9.81 * h)) / 9.81
            distance = v_syn * np.cos(theta_v) * t_flight
        else:
            distance = 0

        # エネルギー計算
        # # 各リンクのエネルギーを計算
        # (kinetic_energy_upper, potential_energy_upper,
        #  kinetic_energy_forearm, potential_energy_forearm,
        #  kinetic_energy_hand, potential_energy_hand) = compute_energies(
        #     data, upper_arm_body_id, forearm_body_id, site_id, m1, m2, m_hand)


        # 報酬の計算
        reward = compute_reward(data, site_id, cumulative_energy)
        # 報酬の記録
        episode_rewards.append(reward)

        # 最大報酬を記録
        if reward > max_reward:
            max_reward = reward
            release_step = step
            max_q_table = Q.copy()

        # Qテーブルの更新
        Q[q1_bin, q2_bin, q3_bin, q4_bin, q1_dot_bin, q2_dot_bin, q3_dot_bin, q4_dot_bin, action] += alpha * (
            reward + gamma * np.max(Q[q1_bin, q2_bin, q3_bin, q4_bin, q1_dot_bin, q2_dot_bin, q3_dot_bin, q4_dot_bin, :]) -
            Q[q1_bin, q2_bin, q3_bin, q4_bin, q1_dot_bin, q2_dot_bin, q3_dot_bin, q4_dot_bin, action]
        )

        # # データを収集
        # episode_data.append([
        #     q1, q2, q3, q4,  # 各自由度の角度
        #     q1_dot, q2_dot, q3_dot, q4_dot,  # 各自由度の角速度
        #     theta_v,  # 投射角度
        #     hand_tip_position[2],  # 手先のz座標（高さ）
        #     vx, vy, vz,  # 各手先速度
        #     v_syn,  # 手先合成速度
        #     shoulder_p_energy_consumed, shoulder_r_energy_consumed, shoulder_y_energy_consumed, elbow_p_energy_consumed,  # 各自由度の消費エネルギー
        #     cumulative_energy,  # 累積消費エネルギー
        #     kinetic_energy_upper, potential_energy_upper,  # 上腕のエネルギー
        #     kinetic_energy_forearm, potential_energy_forearm,  # 前腕のエネルギー
        #     kinetic_energy_hand, potential_energy_hand,  # 手先のエネルギー
        #     torque  # トルク
        # ])


        # 状態の更新
        state = next_state

    # # リリースステップまでの累積報酬和を計算
    # cumulative_reward_until_release = sum(episode_rewards[:release_step + 1]) if release_step != -1 else 0

    # 真の報酬を再計算
    if max_q_table is not None:
        true_reward = compute_true_reward(max_q_table, digitize_state(*state))
        if true_reward > max_true_reward:
            max_true_reward = true_reward
            best_q_table = max_q_table.copy()
    else:
        true_reward = 0

    # ターミナル出力
    print(f"Episode {episode + 1}/{num_episodes}, Max Reward: {max_reward:.3f}, Max Step: {release_step}, True Reward: {true_reward:.3f}")

 
    # CSVにエピソードデータを保存
    # episode_df = pd.DataFrame(episode_data, columns=[
        # # 'q1', 'q2', 'q3', 'q4', 'q1_dot', 'q2_dot', 'q3_dot', 'q4_dot',
        # 'theta_v', 'z_position', 'vx', 'vy', 'vz', 'v_syn',
        # 'shoulder_pitch_energy', 'shoulder_roll_energy', 'shoulder_yaw_energy', 'elbow_pitch_energy',
        # 'cumulative_energy',
        # 'kinetic_energy_upper', 'potential_energy_upper',
        # 'kinetic_energy_forearm', 'potential_energy_forearm',
        # 'kinetic_energy_hand', 'potential_energy_hand',
        # 'torque'
    # ])
    # episode_df.to_csv(f"{info_dir}/episode_{episode + 1}.csv", index=False)

    # 報酬履歴を記録
    reward_progress.append([episode + 1, max_reward, release_step, true_reward])

    # 1エピソードごとに最大True RewardのQテーブルを保存
    if (episode + 1) % 1 == 0:
        if release_step != -1:
            np.save(f"{qtable_dir}/Q_max_reward_episode_{episode + 1}.npy", best_q_table)
            print(f"Saved Q table for episodes_{episode + 1}, with Max True Reward: {max_true_reward:.3f}")
        else:
            np.save(f"{qtable_dir}/Q_max_true_reward_episode_{episode + 1}_no_release.npy", best_q_table)
            print(f"Saved Q table for episodes_{episode + 1}, No Release Step, with Max True Reward: {max_true_reward:.3f}")
        max_true_reward = -float('inf')  # 次の1000エピソード用にリセット
        best_q_table = None  # リセット


    # 結果をCSV保存
    reward_df = pd.DataFrame(reward_progress, columns=['Episode', 'Max Reward', 'Max Step', 'True Reward'])
    reward_df.to_csv(f"{reward_dir}/reward_progress_1.72_0.14.csv", index=False)
