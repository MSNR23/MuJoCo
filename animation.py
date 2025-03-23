import mujoco
import numpy as np
import pandas as pd
import cv2

# モデルとデータの読み込み
model_path = "g1.xml"
csv_path = "1.72_0.14.csv"
output_video = "1.72_0.14.mp4"

# MuJoCo モデルを読み込み
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)

# カメラの設定（mjvCamera を使用）
camera = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(camera)
camera.lookat[:] = [0.0, 0.0, 0.2]  # 注視点を設定
camera.distance = 1.5  # 遠ざける
camera.azimuth = 90 #   見る角度
camera.elevation = 0  # 俯角を調整（上から見下ろす）

# CSVデータを読み込む（最初の491行のみ）
csv_data = pd.read_csv(csv_path).iloc[:449]

# シミュレーション設定
duration = 4.49  # 動画の長さ (秒)
extra_duration = 3.42 # 最終フレームをキープする時間 (秒)
framerate = len(csv_data) / duration  # フレームレートを計算
scene_option = mujoco.MjvOption()  # シーンオプション
frames = []

# シミュレーション実行
mujoco.mj_resetData(model, data)
for i in range(len(csv_data)):
    data.qpos[:4] = csv_data.iloc[i][["q1", "q2", "q3", "q4"]].values
    mujoco.mj_forward(model, data)
    mujoco.mj_step(model, data)

    # シーンを更新してフレームを取得
    renderer.update_scene(data, camera=camera, scene_option=scene_option)  # カメラ適用
    frame = renderer.render()  # フレーム取得 (上下反転しない)
    frames.append(frame)

# **最終フレームを3秒間キープ**
extra_frames = int(extra_duration * framerate)  # 追加するフレーム数
last_frame = frames[-1]  # 最後のフレームを取得

for _ in range(extra_frames):
    frames.append(last_frame)  # 最後のフレームを追加

# OpenCV で動画を保存
height, width, _ = frames[0].shape
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(output_video, fourcc, int(framerate), (width, height))

for frame in frames:
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # OpenCV形式に変換
    video_writer.write(frame_bgr)

video_writer.release()
print(f"動画が {output_video} に保存されました。")
