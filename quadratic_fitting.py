import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def fit_quadratic_2d(p, r, z):
    """
    p(pitch), r(roll)를 독립변수로 하여 z 데이터에 대한 2차 다항식 계수를 구합니다.
    수식: z = c0 + c1*p + c2*r + c3*p^2 + c4*r^2 + c5*p*r
    """
    # 1(상수항), p, r, p^2, r^2, p*r 로 구성된 디자인 행렬(Design Matrix) 생성
    A = np.vstack([np.ones_like(p), p, r, p**2, r**2, p * r]).T
    
    # 최소제곱법(Least-Squares) 계산
    coeffs, residuals, rank, s = np.linalg.lstsq(A, z, rcond=None)
    return coeffs

def calculate_metrics(actual, predicted):
    """오차(편차) 지표 계산"""
    errors = actual - predicted
    rmse = np.sqrt(np.mean(errors**2))      # 루트 평균 제곱 오차
    max_err = np.max(np.abs(errors))        # 최대 절대 편차
    mae = np.mean(np.abs(errors))           # 평균 절대 편차
    return rmse, max_err, mae

def main():
    # 1. 데이터 불러오기
    csv_filename = "pitch_roll_ball_log.csv"
    try:
        df = pd.read_csv(csv_filename)
    except FileNotFoundError:
        print(f"Error: '{csv_filename}' 파일이 현재 경로에 없습니다.")
        return

    # 단위 선택 (이전 매핑에 맞춰 degree 단위를 사용하도록 설정, 필요시 _rad로 변경 가능)
    p = df['pitch_deg'].values
    r = df['roll_deg'].values
    bx = df['ball_x'].values
    by = df['ball_y'].values

    # 2. 2차 다항식 피팅 수행
    c_x = fit_quadratic_2d(p, r, bx)  # ball_x에 대한 계수
    c_y = fit_quadratic_2d(p, r, by)  # ball_y에 대한 계수

    # 3. 피팅 결과 모델을 기반으로 예측값 계산
    A = np.vstack([np.ones_like(p), p, r, p**2, r**2, p * r]).T
    pred_bx = A @ c_x
    pred_by = A @ c_y

    # 4. 편차 및 오차 지표 계산
    rmse_x, max_x, mae_x = calculate_metrics(bx, pred_bx)
    rmse_y, max_y, mae_y = calculate_metrics(by, pred_by)

    # 5. 결과 콘솔 출력
    print("=" * 60)
    print(" [1] Ball X 피팅 결과 수식 (Coefficients)")
    print(f"  ball_x = {c_x[0]:.6f} + ({c_x[1]:.6f})*p + ({c_x[2]:.6f})*r")
    print(f"           + ({c_x[3]:.6f})*p^2 + ({c_x[4]:.6f})*r^2 + ({c_x[5]:.6f})*p*r")
    print("      * 편차 지표 (Residual Metrics):")
    print(f"        - RMSE (정밀도 표준편차): {rmse_x:.6f}")
    print(f"        - MAE  (평균 절대 편차): {mae_x:.6f}")
    print(f"        - MAX  (최대 절대 편차): {max_x:.6f}")
    print("-" * 60)
    print(" [2] Ball Y 피팅 결과 수식 (Coefficients)")
    print(f"  ball_y = {c_y[0]:.6f} + ({c_y[1]:.6f})*p + ({c_y[2]:.6f})*r")
    print(f"           + ({c_y[3]:.6f})*p^2 + ({c_y[4]:.6f})*r^2 + ({c_y[5]:.6f})*p*r")
    print("      * 편차 지표 (Residual Metrics):")
    print(f"        - RMSE (정밀도 표준편차): {rmse_y:.6f}")
    print(f"        - MAE  (평균 절대 편차): {mae_y:.6f}")
    print(f"        - MAX  (최대 절대 편차): {max_y:.6f}")
    print("=" * 60)

    # 6. Matplotlib 3D 시각화
    fig = plt.figure(figsize=(14, 6))
    
    # 그리드 데이터 생성 (곡면 시각화용)
    p_space = np.linspace(p.min(), p.max(), 30)
    r_space = np.linspace(r.min(), r.max(), 30)
    P_grid, R_grid = np.meshgrid(p_space, r_space)
    
    A_grid = np.vstack([
        np.ones_like(P_grid.flatten()), 
        P_grid.flatten(), R_grid.flatten(), 
        P_grid.flatten()**2, R_grid.flatten()**2, 
        P_grid.flatten() * R_grid.flatten()
    ]).T
    
    Z_bx_grid = (A_grid @ c_x).reshape(P_grid.shape)
    Z_by_grid = (A_grid @ c_y).reshape(P_grid.shape)

    # 그래프 1: Ball X Fit
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.scatter(p, r, bx, color='blue', alpha=0.4, label='Actual Data')
    ax1.plot_surface(P_grid, R_grid, Z_bx_grid, color='cyan', alpha=0.4, edgecolor='none')
    ax1.set_title('Quadratic Fit for Ball X')
    ax1.set_xlabel('Pitch (deg)')
    ax1.set_ylabel('Roll (deg)')
    ax1.set_zlabel('Ball X Position')
    ax1.legend()

    # 그래프 2: Ball Y Fit
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.scatter(p, r, by, color='red', alpha=0.4, label='Actual Data')
    ax2.plot_surface(P_grid, R_grid, Z_by_grid, color='orange', alpha=0.4, edgecolor='none')
    ax2.set_title('Quadratic Fit for Ball Y')
    ax2.set_xlabel('Pitch (deg)')
    ax2.set_ylabel('Roll (deg)')
    ax2.set_zlabel('Ball Y Position')
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()