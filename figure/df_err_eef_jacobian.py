#!/usr/bin/env python3
"""3D visualization: df_err_i vs eef_centroid velocity (vx, vy) with fitted plane.

Shows the 6 partial derivatives d(df_err_i)/d(eef_centroid_velocity_j) as plane slopes,
grouped into 3 subplots (one per force sensor).
"""

import csv
import numpy as np
import matplotlib.pyplot as plt


def load_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows


def fit_plane(X, Y, Z):
    """Fit Z = a*X + b*Y + c, return (a, b, c)."""
    A = np.column_stack([X, Y, np.ones_like(X)])
    coeff, *_ = np.linalg.lstsq(A, Z, rcond=None)
    return coeff[0], coeff[1], coeff[2]


def plot_one(ax, X, Y, Z, title, cmap='viridis'):
    ax.scatter(X, Y, Z, c=Z, cmap=cmap, s=2, alpha=0.6)

    a, b, c = fit_plane(X, Y, Z)
    x_range = np.linspace(X.min(), X.max(), 20)
    y_range = np.linspace(Y.min(), Y.max(), 20)
    xx, yy = np.meshgrid(x_range, y_range)
    zz = a * xx + b * yy + c
    ax.plot_surface(xx, yy, zz, alpha=0.35, color='red')

    ax.set_xlabel('eef_centroid_vx')
    ax.set_ylabel('eef_centroid_vy')
    ax.set_zlabel('df_err')
    ax.set_title(f'{title}\nd(df)/dvx={a:.2f}  d(df)/dvy={b:.2f}')


def main():
    rows = load_csv('ball_balancing_log.csv')

    timestamps = np.array([float(r['timestamp']) for r in rows])
    X = np.array([float(r['eef_centroid_x']) for r in rows])
    Y = np.array([float(r['eef_centroid_y']) for r in rows])

    # finite-difference centroid velocity
    dt = np.diff(timestamps)
    vx = np.diff(X) / np.maximum(dt, 1e-9)
    vy = np.diff(Y) / np.maximum(dt, 1e-9)

    # df_err from CSV (trim first row to match diff length)
    DF1 = np.array([float(r['df_err1']) for r in rows[1:]])
    DF2 = np.array([float(r['df_err2']) for r in rows[1:]])
    DF3 = np.array([float(r['df_err3']) for r in rows[1:]])

    fig = plt.figure(figsize=(18, 5.5))

    for idx, (DF, title) in enumerate([
        (DF1, 'df_err1 (sensor 1)'),
        (DF2, 'df_err2 (sensor 2)'),
        (DF3, 'df_err3 (sensor 3)'),
    ]):
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
        plot_one(ax, vx, vy, DF, title)

    fig.tight_layout()
    plt.savefig('figure/df_err_eef_jacobian.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
