import mujoco
import mujoco.viewer as viewer
import numpy as np
import time

# 修正後のXMLファイルのパス
model_path = "g1.xml"

# MuJoCoモデルをロード
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# 各アクチュエータ名を取得（肩ピッチ、肩ロール、肩ヨー、肘）
actuator_names = ["shoulder_pitch_actuator", "shoulder_roll_actuator", "shoulder_yaw_actuator", "elbow_actuator"]

# アクチュエータ名からインデックスを取得
actuator_indices = []
for name in actuator_names:
    for i in range(model.nu):
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) == name:
            actuator_indices.append(i)
            break
if len(actuator_indices) != len(actuator_names):
    raise ValueError("Some actuators not found in the model.")

# 各自由度のトルク範囲を取得
torque_limits = model.actuator_ctrlrange[actuator_indices]

# 4自由度にランダムトルクを適用する関数
def apply_random_torques_to_all(data):
    """
    肩ピッチ、肩ロール、肩ヨー、肘にランダムなトルクを適用
    """
    random_torques = np.random.uniform(torque_limits[:, 0], torque_limits[:, 1])
    data.ctrl[:] = 0  # 全トルクを初期化
    for idx, torque in zip(actuator_indices, random_torques):
        data.ctrl[idx] = torque  # 各自由度にランダムトルクを適用
    print(f"Applied Random Torques: {random_torques}")

# ビューアを起動
with viewer.launch_passive(model, data) as sim_viewer:
    print("MuJoCo Viewer launched in passive mode.")
    print("Press Ctrl+C to exit the viewer...")

    try:
        while sim_viewer.is_running():
            apply_random_torques_to_all(data)  # 4自由度すべてにランダムトルクを適用
            mujoco.mj_step(model, data)  # シミュレーションを1ステップ進める
            sim_viewer.sync()  # ビューアを同期
            time.sleep(0.02)  # シミュレーションステップ時間
    except KeyboardInterrupt:
        print("Exiting Viewer.")
