import mujoco
import mujoco_viewer
import numpy as np
import time
import csv
from scipy.interpolate import RectBivariateSpline
import pandas as pd

# モデルのパス
model_path = "hougan.xml"

# MuJoCoモデルの読み込み
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# シミュレーション設定
dt = model.opt.timestep  # タイムステップ
num_steps = int(5 / dt)  # 5秒で終わるようにステップ数を設定
frame_interval = 5 / num_steps  # フレーム間隔（5秒で終了）

# トルクを適用するアクチュエータの順番
actuator_names = [
    "shoulder_pitch_actuator",
    "shoulder_roll_actuator",
    "shoulder_yaw_actuator",
    "elbow_actuator",
]

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

# 肩ヨーの可動範囲を取得する関数
def get_yaw_range(pitch, roll):
    yaw_max = yaw_max_spline(pitch, roll)[0][0]  # 補間された最大ヨー角
    yaw_min = yaw_min_spline(pitch, roll)[0][0]  # 補間された最小ヨー角
    return yaw_min, yaw_max

# ランダムトルク生成関数（肩ヨーに動的制約を追加）
def generate_random_torque():
    # 各アクチュエータのランダムトルクを生成
    shoulder_pitch = np.random.choice([0, 0, 0, 20, 40])
    shoulder_roll = np.random.choice([0, 0, 0, 20, 40])
    
    # 肩ヨーの可動範囲を動的に取得
    yaw_min, yaw_max = get_yaw_range(shoulder_pitch, shoulder_roll)
    shoulder_yaw = np.random.uniform(yaw_min, yaw_max)  # 制約された範囲でランダム生成

    elbow = np.random.choice([-30, -15, 0, 15, 30])
    
    return [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow]

# Viewerのセットアップ
viewer = mujoco_viewer.MujocoViewer(model, data)
is_paused = False  # 一時停止フラグ
pause_start_time = None  # 一時停止の開始時間
current_step = 0  # 現在のステップを追跡

# 角度データ保存用
angle_data = []

try:
    while current_step < num_steps:
        # 一時停止中の処理
        if is_paused:
            if pause_start_time is None:
                pause_start_time = time.time()  # 一時停止開始時刻を記録
            time.sleep(0.1)
            if not viewer.render():  # ビューアが閉じられたら終了
                break
            key = viewer.get_last_key()
            if key == ord(" "):  # スペースキーで再開
                is_paused = False
                pause_start_time = None  # リセット
                print("Resumed")
            continue  # シミュレーションを進めない

        # 目標時間（5秒で終わるように調整）
        target_time = time.time() + frame_interval

        # ランダムトルクを生成
        data.ctrl[:] = generate_random_torque()

        # シミュレーションステップを進める
        mujoco.mj_step(model, data)

        # 角度データの保存
        angle_data.append([data.time] + [data.qpos[i] for i in range(model.nq)])

        # ステップを進める
        current_step += 1

        # フレームの描画
        while time.time() < target_time:
            if not viewer.render():  # ビューアが閉じられたら終了
                break

            # 経過時間をオーバーレイに表示
            elapsed_time = current_step * dt
            viewer.add_overlay(mujoco_viewer.GLFW_KEY_NONE, "Elapsed Time", f"{elapsed_time:.2f} sec")

            # スペースキーで一時停止
            key = viewer.get_last_key()
            if key == ord(" "):  
                is_paused = True
                print("Paused")
                break

finally:
    # 角度データをCSVに保存
    with open("angle_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time"] + [f"qpos_{i}" for i in range(model.nq)])  # ヘッダー
        writer.writerows(angle_data)
    
    viewer.close()
