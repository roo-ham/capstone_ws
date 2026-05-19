#!/usr/bin/python3
"""Plot roll angle and motor currents from ball_balancing_log.csv as time series."""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'ball_balancing_log.csv'
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    t = df['timestamp'] - df['timestamp'].iloc[0]
    roll_deg = df['roll_rad'].apply(lambda r: r * 57.2958)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle('Tilt Angle vs Motor Current', fontsize=14, fontweight='bold')

    # --- Subplot 1: Roll angle ---
    ax1.plot(t, roll_deg, color='darkblue', linewidth=0.8)
    ax1.set_ylabel('Roll Angle [deg]', color='darkblue')
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.4)
    ax1.grid(True, alpha=0.3)
    ax1.legend(['Roll'], loc='upper right')

    # --- Subplot 2: Motor currents ---
    ax2.plot(t, df['eef_cur_1'], linewidth=0.8, label='Motor 1')
    ax2.plot(t, df['eef_cur_2'], linewidth=0.8, label='Motor 2')
    ax2.plot(t, df['eef_cur_3'], linewidth=0.8, label='Motor 3')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Current [A]')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    out_png = csv_path.replace('.csv', '_current_vs_tilt.png')
    fig.savefig(out_png, dpi=150)
    print(f"Saved: {out_png}")
    plt.show()


if __name__ == '__main__':
    main()
