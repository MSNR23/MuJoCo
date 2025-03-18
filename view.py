import numpy as np
import mujoco
from mujoco import viewer

# モデルファイルのパス
model_path = "1.90.xml"

# 初期姿勢の設定（角度は度単位で指定）
initial_pose = {
    "right_shoulder_pitch_joint": -1.78985*180/np.pi,  # 初期値: 0度
    "right_shoulder_roll_joint": -2.36152*180/np.pi,   # 初期値: 0度
    "right_shoulder_yaw_joint": 1.142475*180/np.pi,    # 初期値: 0度
    "right_elbow_joint": -0.14144*180/np.pi,          # 肘を90度に設定
    # "right_wrist_roll_joint": 90.0           # 肘を90度に設定
}

# モデルとシミュレーションデータを読み込む
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# 初期姿勢を設定する関数
def set_initial_pose(model, data, initial_pose):
    for joint_name, angle in initial_pose.items():
        joint_id = model.joint(joint_name).qposadr  # qposのインデックス取得
        data.qpos[joint_id] = np.radians(angle)  # 度をラジアンに変換して設定

# 初期姿勢を設定
set_initial_pose(model, data, initial_pose)

# # ビューアの起動
# with viewer.launch_passive(model, data) as viewer_instance:
#     while True:
#         # キーボードやマウスでモデルを操作可能
#         if not viewer_instance.is_running():
#             break

# ビューア起動時にカメラの位置を設定
with viewer.launch_passive(model, data) as viewer_instance:
    # カメラの位置と向きを設定（前方から）
    viewer_instance.cam.lookat[:] = [0, 0, 0.1]  # 視点（モデルの中心）
    viewer_instance.cam.distance = 2.0  # カメラとの距離
    viewer_instance.cam.elevation = 0  # 上下の角度
    # viewer_instance.cam.azimuth = 90  # 左右の角度（正面から）
    viewer_instance.cam.azimuth = 180  # 左右の角度（正面から）

    while True:
        if not viewer_instance.is_running():
            break