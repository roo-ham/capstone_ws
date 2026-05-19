#!/usr/bin/env python3
"""
9-panel 3D visualization: f_err / df_err vs eef centroid (position / velocity).

Row 1: f_err scatter only (no plane)
Row 2: f_err + fitted plane  → stiffness Jacobian:  ∂(f_err)/∂x,  ∂(f_err)/∂y
Row 3: df_err + fitted plane → damping Jacobian:  ∂(df)/∂vx, ∂(df)/∂vy
"""

import csv
import numpy as np
import matplotlib.pyplot as plt


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fit_plane(X, Y, Z):
    A = np.column_stack([X, Y, np.ones_like(X)])
    coeff, *_ = np.linalg.lstsq(A, Z, rcond=None)
    return coeff[0], coeff[1], coeff[2]


def add_plane(ax, X, Y, Z):
    a, b, c = fit_plane(X, Y, Z)
    xr = np.linspace(X.min(), X.max(), 20)
    yr = np.linspace(Y.min(), Y.max(), 20)
    xx, yy = np.meshgrid(xr, yr)
    zz = a * xx + b * yy + c
    ax.plot_surface(xx, yy, zz, alpha=0.35, color='red')
    return a, b


def scatter3d(ax, X, Y, Z, cmap='coolwarm'):
    return ax.scatter(X, Y, Z, c=Z, cmap=cmap, s=1.5, alpha=0.7)


def set_labels(ax, xl, yl, zl, title):
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_zlabel(zl)
    ax.set_title(title, fontsize=9)


def main():
    rows = load_csv('ball_balancing_log.csv')

    timestamps = np.array([float(r['timestamp']) for r in rows])
    X = np.array([float(r['eef_centroid_x']) for r in rows])
    Y = np.array([float(r['eef_centroid_y']) for r in rows])
    F1 = np.array([float(r['f_err1']) for r in rows])
    F2 = np.array([float(r['f_err2']) for r in rows])
    F3 = np.array([float(r['f_err3']) for r in rows])

    # centroid velocity (finite difference)
    dt = np.diff(timestamps)
    vx = np.diff(X) / np.maximum(dt, 1e-9)
    vy = np.diff(Y) / np.maximum(dt, 1e-9)
    DF1 = np.array([float(r['df_err1']) for r in rows[1:]])
    DF2 = np.array([float(r['df_err2']) for r in rows[1:]])
    DF3 = np.array([float(r['df_err3']) for r in rows[1:]])

    sensors = [(F1, F2, F3), (DF1, DF2, DF3)]
    titles_f = ['f_err1 (sensor 1)', 'f_err2 (sensor 2)', 'f_err3 (sensor 3)']
    titles_df = ['df_err1 (sensor 1)', 'df_err2 (sensor 2)', 'df_err3 (sensor 3)']

    fig = plt.figure(figsize=(16, 14))

    # ── Row 1: f_err scatter only ──
    for i, (F, tl) in enumerate(zip([F1, F2, F3], titles_f)):
        ax = fig.add_subplot(3, 3, i + 1, projection='3d')
        scatter3d(ax, X, Y, F)
        set_labels(ax, 'centroid_x', 'centroid_y', 'f_err', f'[Scatter] {tl}')

    # ── Row 2: f_err + plane (stiffness) ──
    for i, (F, tl) in enumerate(zip([F1, F2, F3], titles_f)):
        ax = fig.add_subplot(3, 3, 4 + i, projection='3d')
        scatter3d(ax, X, Y, F)
        a, b = add_plane(ax, X, Y, F)
        set_labels(ax, 'centroid_x', 'centroid_y', 'f_err',
                   f'{tl}\n∂f/∂x={a:.2f}  ∂f/∂y={b:.2f}')

    # ── Row 3: df_err + plane (damping) ──
    for i, (DF, tl) in enumerate(zip([DF1, DF2, DF3], titles_df)):
        ax = fig.add_subplot(3, 3, 7 + i, projection='3d')
        scatter3d(ax, vx, vy, DF)
        a, b = add_plane(ax, vx, vy, DF)
        set_labels(ax, 'centroid_vx', 'centroid_vy', 'df_err',
                   f'{tl}\n∂(df)/∂vx={a:.2f}  ∂(df)/∂vy={b:.2f}')

    fig.tight_layout(pad=2)
    plt.savefig('figure/9panel_f_df_eef_jacobian.png', dpi=150)
    print('Saved: figure/9panel_f_df_eef_jacobian.png')
    plt.show()


if __name__ == '__main__':
    main()
