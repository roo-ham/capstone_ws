import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def fit_2nd_order_surface(x, y, z):
    """
    x, y 좌표와 z 값에 대해 2차 다항식 곡면 피팅을 수행합니다.
    모델: z = c0 + c1*x + c2*y + c3*x^2 + c4*y^2 + c5*xy
    """
    # 디자인 행렬(Design Matrix) 생성
    A = np.c_[np.ones(x.shape), x, y, x**2, y**2, x*y]
    
    # 최소제곱법으로 계수(Coefficients) 계산
    C, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    return C

def predict_surface(x, y, C):
    """구해진 계수를 바탕으로 z 값을 예측합니다."""
    return C[0] + C[1]*x + C[2]*y + C[3]*(x**2) + C[4]*(y**2) + C[5]*(x*y)

# 1. 데이터 로드 및 샘플링
filename = 'ball_balancing_log.csv'
if not os.path.exists(filename):
    print(f"Error: '{filename}' 파일이 없습니다.")
    exit()

df = pd.read_csv(filename)

# 데이터가 너무 많으면 렌더링이 느려지므로 최대 1000개만 무작위 샘플링
num_samples = 1000
if len(df) > num_samples:
    df_sampled = df.sample(n=num_samples, random_state=42)
else:
    df_sampled = df

# Radian을 Degree로 변환
x_data = np.degrees(df_sampled['roll_rad'].values)
y_data = np.degrees(df_sampled['pitch_rad'].values)
f_data = [df_sampled['f1_raw'].values, df_sampled['f2_raw'].values, df_sampled['f3_raw'].values]

# 곡면(Surface)을 그리기 위한 격자(Grid) 생성
x_grid = np.linspace(x_data.min(), x_data.max(), 30)
y_grid = np.linspace(y_data.min(), y_data.max(), 30)
X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)

# 2. 시각화 준비
fig = plt.figure(figsize=(18, 6))
fig.suptitle('2nd-Order Polynomial Surface Fitting (1000 Samples)', fontsize=16)

# 3. 3개의 센서 각각에 대해 피팅 및 플로팅
for i in range(3):
    z_data = f_data[i]
    
    # 피팅 수행
    coeffs = fit_2nd_order_surface(x_data, y_data, z_data)
    print(f"--- Sensor {i+1} Coefficients ---")
    print(f"c0 (상수):  {coeffs[0]:.4f}")
    print(f"c1 (x):     {coeffs[1]:.4f}")
    print(f"c2 (y):     {coeffs[2]:.4f}")
    print(f"c3 (x^2):   {coeffs[3]:.4f}")
    print(f"c4 (y^2):   {coeffs[4]:.4f}")
    print(f"c5 (xy):    {coeffs[5]:.4f}\n")
    
    # 생성한 격자에 대한 피팅 곡면 Z 값 계산
    Z_mesh = predict_surface(X_mesh, Y_mesh, coeffs)
    
    # 3D Plot
    ax = fig.add_subplot(1, 3, i+1, projection='3d')
    
    # 원본 샘플 데이터 (Scatter) - 투명도를 주어 표면과 겹쳐 보이게 함
    ax.scatter(x_data, y_data, z_data, color='k', s=10, alpha=0.3, label='Sampled Data')
    
    # 피팅된 곡면 (Surface)
    surf = ax.plot_surface(X_mesh, Y_mesh, Z_mesh, cmap='viridis', alpha=0.8, edgecolor='none')
    
    ax.set_title(f'Sensor {i+1} Fitting')
    ax.set_xlabel('Roll (deg)')
    ax.set_ylabel('Pitch (deg)')
    ax.set_zlabel(f'Force {i+1}')
    ax.legend()
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

plt.tight_layout()
plt.show()