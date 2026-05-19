#!/usr/bin/python3
"""Plot roll/pitch angle vs tactile sensor force with linear fitting and correlation analysis."""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

SENSOR_COLORS = ['#e74c3c', '#2ecc71', '#3498db']  # red, green, blue


def linear_fit(x, y):
    """Return (a, b, r2, r) for y = a*x + b."""
    mask = np.isfinite(x) & np.isfinite(y)
    x_m, y_m = x[mask], y[mask]
    if len(x_m) < 3:
        return 0.0, 0.0, 0.0, 0.0
    a, b = np.polyfit(x_m, y_m, 1)
    y_pred = a * x_m + b
    ss_res = np.sum((y_m - y_pred) ** 2)
    ss_tot = np.sum((y_m - np.mean(y_m)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    r = np.corrcoef(x_m, y_m)[0, 1]
    return a, b, r2, r


def plot_scatter_with_fit(ax, x_data, y_cols, df, xlabel, title):
    """Scatter + rolling mean + linear fit per sensor on given axes."""
    x = x_data.values
    for i, col in enumerate(y_cols):
        y = df[col].values
        ax.scatter(x, y, s=2, color=SENSOR_COLORS[i], alpha=0.4, rasterized=True)

        # rolling mean trend
        sorted_idx = np.argsort(x)
        window = max(50, len(df) // 100)
        trend = pd.Series(y[sorted_idx]).rolling(window=window, center=True, min_periods=10).mean()
        ax.plot(x[sorted_idx], trend, color=SENSOR_COLORS[i], linewidth=1.0, alpha=0.7)

        # linear fit line
        a, b, r2, r = linear_fit(x, y)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, a * x_line + b, color=SENSOR_COLORS[i], linewidth=2, linestyle='--',
                label=f'S{i+1}: y={a:.2f}x+{b:.1f}, R²={r2:.4f}, r={r:.4f}')

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Force Residual [gf]')
    ax.set_title(title)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.4)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.4)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=7)


def print_correlation_table(df):
    """Print Pearson correlation matrix between tilt axes and force sensors."""
    cols = ['roll_rad', 'pitch_rad', 'f1_res', 'f2_res', 'f3_res']
    labels = ['Roll', 'Pitch', 'F1_res', 'F2_res', 'F3_res']
    data = df[cols].dropna()
    corr = np.corrcoef(data.values.T)

    print("\n" + "=" * 70)
    print("Pearson Correlation Matrix (r)")
    print("=" * 70)
    header = f"{'':>12}" + "".join(f"{l:>10}" for l in labels)
    print(header)
    print("-" * 62)
    for i, li in enumerate(labels):
        row = f"{li:>12}" + "".join(f"{corr[i,j]:>10.4f}" for j in range(len(labels)))
        print(row)

    print("\n[Key Correlations - Tilt vs Force]")
    for ti, tname in [(0, 'Roll'), (1, 'Pitch')]:
        for fi, fname in [(2, 'F1'), (3, 'F2'), (4, 'F3')]:
            print(f"  {tname:5s} x {fname:5s}: r = {corr[ti, fi]:+.4f}")

    print("\n" + "=" * 70)
    print("Linear Fit Summary: force = a * angle_deg + b")
    print("=" * 70)
    print(f"{'Pair':>20s}  {'a [gf/deg]':>12s}  {'b [gf]':>10s}  {'R²':>10s}  {'r':>10s}")
    print("-" * 68)

    roll_deg = np.degrees(data['roll_rad'])
    pitch_deg = np.degrees(data['pitch_rad'])

    for tname, x in [('Roll', roll_deg), ('Pitch', pitch_deg)]:
        for fi, fname in enumerate(['F1_res', 'F2_res', 'F3_res'], start=1):
            y = data[f'f{fi}_res'].values
            a, b, r2, r = linear_fit(x.values, y)
            pair = f"{tname} -> Sensor {fi}"
            print(f"{pair:>20s}  {a:>12.4f}  {b:>10.2f}  {r2:>10.4f}  {r:>10.4f}")
    print()


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'ball_balancing_log.csv'
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    t = df['timestamp'] - df['timestamp'].iloc[0]
    roll_deg = np.degrees(df['roll_rad'])
    pitch_deg = np.degrees(df['pitch_rad'])
    f_cols = ['f1_res', 'f2_res', 'f3_res']

    fig, axes = plt.subplots(5, 1, figsize=(14, 16))
    (ax1, ax2, ax3, ax4, ax5) = axes
    fig.suptitle('Roll / Pitch vs Tactile Sensor Force', fontsize=14, fontweight='bold')

    # --- Row 1: Roll angle time series ---
    ax1.plot(t, roll_deg, color='darkblue', linewidth=0.8)
    ax1.set_ylabel('Roll [deg]')
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.4)
    ax1.grid(True, alpha=0.3)

    # --- Row 2: Pitch angle time series ---
    ax2.plot(t, pitch_deg, color='darkred', linewidth=0.8)
    ax2.set_ylabel('Pitch [deg]')
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.4)
    ax2.grid(True, alpha=0.3)

    # --- Row 3: Force residuals time series ---
    for i, col in enumerate(f_cols):
        ax3.plot(t, df[col], color=SENSOR_COLORS[i], linewidth=0.6, alpha=0.85, label=f'Sensor {i+1}')
    ax3.set_ylabel('Force Residual [gf]')
    ax3.axhline(0, color='gray', linestyle='--', alpha=0.4)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right')

    # --- Row 4: f_res vs Roll (2D scatter + fits) ---
    plot_scatter_with_fit(ax4, roll_deg, f_cols, df, 'Roll Angle [deg]', 'Force Residual vs Roll')

    # --- Row 5: f_res vs Pitch (2D scatter + fits) ---
    plot_scatter_with_fit(ax5, pitch_deg, f_cols, df, 'Pitch Angle [deg]', 'Force Residual vs Pitch')

    # --- Correlation analysis ---
    print_correlation_table(df)

    plt.tight_layout()
    out_png = csv_path.replace('.csv', '_force_vs_tilt.png')
    fig.savefig(out_png, dpi=150)
    print(f"Saved: {out_png}")
    plt.show()


if __name__ == '__main__':
    main()
