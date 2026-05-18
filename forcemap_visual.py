import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. CSV 파일 로드
filename = 'force_residue_log.csv'

if not os.path.exists(filename):
    print(f"Error: '{filename}' 파일이 존재하지 않습니다.")
    exit()

df = pd.read_csv(filename)

# 2. 데이터 추출 및 단위 변환 (Radian -> Degree)
roll_deg = df['target_x']
pitch_deg = df['target_y']
f1 = df['f1_res']
f2 = df['f2_res']
f3 = df['f3_res']

# 3. 그래프 창 설정 (가로로 길게 3개의 3D Plot 배치)
fig = plt.figure(figsize=(18, 6))
fig.suptitle('Force Sensor 3D Mapping (Roll & Pitch vs Force)', fontsize=16)

# ----------------- 첫 번째 맵 (Force 1) -----------------
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
sc1 = ax1.scatter(roll_deg, pitch_deg, f1, c=f1, cmap='viridis', marker='o', alpha=0.6)
ax1.set_title('Sensor 1 (f1)')
ax1.set_xlabel('Roll (deg)')
ax1.set_ylabel('Pitch (deg)')
ax1.set_zlabel('Force 1')
fig.colorbar(sc1, ax=ax1, shrink=0.5, aspect=10, label='Force Magnitude')

# ----------------- 두 번째 맵 (Force 2) -----------------
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
sc2 = ax2.scatter(roll_deg, pitch_deg, f2, c=f2, cmap='plasma', marker='o', alpha=0.6)
ax2.set_title('Sensor 2 (f2)')
ax2.set_xlabel('Roll (deg)')
ax2.set_ylabel('Pitch (deg)')
ax2.set_zlabel('Force 2')
fig.colorbar(sc2, ax=ax2, shrink=0.5, aspect=10, label='Force Magnitude')

# ----------------- 세 번째 맵 (Force 3) -----------------
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
sc3 = ax3.scatter(roll_deg, pitch_deg, f3, c=f3, cmap='inferno', marker='o', alpha=0.6)
ax3.set_title('Sensor 3 (f3)')
ax3.set_xlabel('Roll (deg)')
ax3.set_ylabel('Pitch (deg)')
ax3.set_zlabel('Force 3')
fig.colorbar(sc3, ax=ax3, shrink=0.5, aspect=10, label='Force Magnitude')

# 4. 레이아웃 조정 및 출력
plt.tight_layout()
plt.show()