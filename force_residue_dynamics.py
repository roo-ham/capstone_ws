import pandas as pd
import numpy as np
import os

# 1. 데이터 로드
filename = 'force_residue_log.csv'
if not os.path.exists(filename):
    print(f"Error: '{filename}' 파일이 존재하지 않습니다.")
    exit()

df = pd.read_csv(filename)

# 데이터 행 수가 미분을 계산하기에 너무 적으면 종료
if len(df) < 5:
    print("데이터가 너무 적어 미분 및 분석이 불가능합니다.")
    exit()

# 2. 독립변수 구성 (Roll/Pitch 각도 및 수치 미분을 통한 각속도 계산)
# 수치 미분 시 노이즈가 튈 수 있으므로 dt 계산에 유의합니다.
dt = df['timestamp'].diff().values
dt[0] = dt[1]  # 첫 번째 행의 NaN 방지 처리

# 각속도 (rad/s) 계산
omega_roll = df['roll_rad'].diff().values / dt
omega_pitch = df['pitch_rad'].diff().values / dt

# 첫 번째 행은 미분값이 불완전하므로 안전하게 제외하고 복사
omega_roll[0] = omega_roll[1]
omega_pitch[0] = omega_pitch[1]

# 독립변수 행렬 X 구성 (N x 5) -> [1, roll, pitch, w_roll, w_pitch] (상수항 포함)
# degree 단위를 원하시면 df['roll_deg']를 쓰셔도 됩니다. 여기서는 물리 단위(rad) 기준입니다.
roll = df['roll_rad'].values
pitch = df['pitch_rad'].values

X = np.c_[np.ones(len(df)), roll, pitch]

# 3. 종속변수 행렬 Y 구성 (N x 3) -> [f1_res, f2_res, f3_res]
Y = df[['f1_res', 'f2_res', 'f3_res']].values

# 4. 다변량 선형 회귀 수행 (최소제곱법)
# X * W = Y  ->  W는 (5 x 3) 크기의 행렬이 됩니다.
W, residuals_sum, rank, s = np.linalg.lstsq(X, Y, rcond=None)

# 5. 모델 기반 예측 값 및 '최종 동적 잔차(Final Dynamic Residue)' 계산
Y_pred = X.dot(W)
final_residuals = Y - Y_pred  # 실제 정적잔차 - 동적모델 예측값

# 6. 통계치 산출 (최대 크기 및 표준편차)
max_error = np.max(np.abs(final_residuals), axis=0)
std_error = np.std(final_residuals, axis=0)
mean_error = np.mean(final_residuals, axis=0)

std = np.std(Y, axis=0)
print(f"  - 오차 표준편차 (Std):  {std}")

# 7. 결과 출력
print("=========================================================")
print("  4차원 입력 (Roll, Pitch, W_roll, W_pitch) -> 3차원 Force 출력")
print("           동적 선형 벡터 함수 회귀 분석 결과")
print("=========================================================\n")

sensor_names = ['Sensor 1 (f1)', 'Sensor 2 (f2)', 'Sensor 3 (f3)']

for i in range(3):
    print(f"[{sensor_names[i]} 최종 잔차 통계]")
    print(f"  - 평균 오차 (Bias):    {mean_error[i]:.4f}")
    print(f"  - 최대 오차 크기 (Max):  {max_error[i]:.4f}")
    print(f"  - 오차 표준편차 (Std):  {std_error[i]:.4f}")
    print(f"  - 매핑 피팅 방정식:")
    print(f"    f_res = {W[0,i]:.4f} + ({W[1,i]:.4f}*roll) + ({W[2,i]:.4f}*pitch)\n")

print("=========================================================")
print("💡 분석 가이드:")
print("1. 표준편차(Std)가 0에 가까울수록 동적 보정이 완벽하게 이루어짐을 뜻합니다.")
print("2. 만약 정적 잔차에 비해 최대 오차(Max)나 표준편차가 획기적으로 줄었다면,")
print("   이 '매핑 피팅 방정식'의 계수(W 행렬)를 C++이나 제어 노드에 추가하여")
print("   동적 힘 보정(Dynamic Force Compensation) 루틴으로 즉시 활용할 수 있습니다.")
print("=========================================================")