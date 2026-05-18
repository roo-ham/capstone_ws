import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def create_design_matrix(f1, f2, f3):
    """
    3개의 변수(f1, f2, f3)에 대한 2차 다항식 디자인 행렬을 생성합니다.
    항 구성 (총 10개): 1, f1, f2, f3, f1^2, f2^2, f3^2, f1*f2, f2*f3, f1*f3
    """
    return np.c_[
        np.ones(f1.shape), 
        f1, f2, f3, 
        f1**2, f2**2, f3**2, 
        f1*f2, f2*f3, f1*f3
    ]

def fit_inverse_model(A, target):
    """
    디자인 행렬 A와 목표값(target)을 이용해 최소제곱법으로 계수를 계산합니다.
    """
    C, _, _, _ = np.linalg.lstsq(A, target, rcond=None)
    return C

def predict_values(A, C):
    """구해진 계수를 바탕으로 목표값을 예측합니다."""
    return A @ C # 행렬 곱셈

# 1. 데이터 로드 및 샘플링
filename = 'force_residue_log.csv'
if not os.path.exists(filename):
    print(f"Error: '{filename}' 파일이 없습니다.")
    exit()

df = pd.read_csv(filename)

# 데이터 샘플링
num_samples = 1000
if len(df) > num_samples:
    df_sampled = df.sample(n=num_samples, random_state=42)
else:
    df_sampled = df

# 독립 변수 (Forces)
f1 = df_sampled['f1_res'].values
f2 = df_sampled['f2_res'].values
f3 = df_sampled['f3_res'].values

# 종속 변수 (Positions)
x_data = df_sampled['target_x'].values
y_data = df_sampled['target_y'].values

# 2. 디자인 행렬 생성 및 피팅
A = create_design_matrix(f1, f2, f3)

# X 추정
coeffs_x = fit_inverse_model(A, x_data)
pred_x = predict_values(A, coeffs_x)

# Y 추정
coeffs_y = fit_inverse_model(A, y_data)
pred_y = predict_values(A, coeffs_y)

# 결과 계수 출력
term_names = ["c0 (상수)", "c1 (f1)", "c2 (f2)", "c3 (f3)", 
              "c4 (f1^2)", "c5 (f2^2)", "c6 (f3^2)", 
              "c7 (f1*f2)", "c8 (f2*f3)", "c9 (f1*f3)"]

print("--- X 추정 2차 다항식 계수 ---")
for name, coef in zip(term_names, coeffs_x):
    print(f"{name:<12}: {coef:.6f}")

print("\n--- Y 추정 2차 다항식 계수 ---")
for name, coef in zip(term_names, coeffs_y):
    print(f"{name:<12}: {coef:.6f}")

# 3. 시각화 (Actual vs Predicted)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Inverse Model: Predicting X, Y from Forces (f1, f2, f3)', fontsize=16)

# X Plot
axes[0].scatter(x_data, pred_x, alpha=0.5, color='blue', edgecolors='k')
axes[0].plot([x_data.min(), x_data.max()], [x_data.min(), x_data.max()], 'r--', lw=2, label='Ideal Fit (y=x)')
axes[0].set_title('Actual vs Predicted Target X')
axes[0].set_xlabel('Actual X')
axes[0].set_ylabel('Predicted X')
axes[0].grid(True, linestyle=':', alpha=0.7)
axes[0].legend()

# Y Plot
axes[1].scatter(y_data, pred_y, alpha=0.5, color='green', edgecolors='k')
axes[1].plot([y_data.min(), y_data.max()], [y_data.min(), y_data.max()], 'r--', lw=2, label='Ideal Fit (y=x)')
axes[1].set_title('Actual vs Predicted Target Y')
axes[1].set_xlabel('Actual Y')
axes[1].set_ylabel('Predicted Y')
axes[1].grid(True, linestyle=':', alpha=0.7)
axes[1].legend()

plt.tight_layout()
plt.show()