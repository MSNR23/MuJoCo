import pandas as pd
import numpy as np
from scipy.interpolate import RectBivariateSpline

# Excelファイルを読み込む
file_path = "yaw_max.xlsx"  # エクセルファイルのパス
df = pd.read_excel(file_path, header=None)

# X軸（1行目の2列目以降）
x = df.iloc[0, 1:].to_numpy()
# Y軸（1列目の2行目以降）
y = df.iloc[1:, 0].to_numpy()
# Z値（B2:F8の部分）
z = df.iloc[1:, 1:].to_numpy()

# スプライン補間の設定
spline = RectBivariateSpline(y, x, z)

# 任意の (x, y) における補間
x_query = -135  # 任意のx
y_query = -135   # 任意のy
z_query = spline(y_query, x_query)  # スプライン補間

print(f"補間結果: z({x_query}, {y_query}) = {z_query[0][0]}")
