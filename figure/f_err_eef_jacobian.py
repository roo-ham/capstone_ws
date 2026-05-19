#!/usr/bin/env python3
"""3D visualization: f_err_i vs eef_centroid (x, y) with fitted plane.

Shows the 6 partial derivatives d(f_err_i)/d(eef_centroid_j) as plane slopes,
grouped into 3 subplots (one per force sensor).
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
    """3D scatter + fitted plane on given axis."""
    ax.scatter(X, Y, Z, c=Z, cmap=cmap, s=2, alpha=0.6)

    # fitted plane mesh
    a, b, c = fit_plane(X, Y, Z)
    x_range = np.linspace(X.min(), X.max(), 20)
    y_range = np.linspace(Y.min(), Y.max(), 20)
    xx, yy = np.meshgrid(x_range, y_range)
    zz = a * xx + b * yy + c
    ax.plot_surface(xx, yy, zz, alpha=0.35, color='red')

    ax.set_xlabel('eef_centroid_x')
    ax.set_ylabel('eef_centroid_y')
    ax.set_zlabel('f_err')
    ax.set_title(f'{title}\ndf/dx={a:.2f}  df/dy={b:.2f}')


def main():
    rows = load_csv('ball_balancing_log.csv')

    X = np.array([float(r['eef_centroid_x']) for r in rows])
    Y = np.array([float(r['eef_centroid_y']) for r in rows])

    F1 = np.array([float(r['f_err1']) for r in rows])
    F2 = np.array([float(r['f_err2']) for r in rows])
    F3 = np.array([float(r['f_err3']) for r in rows])

    fig = plt.figure(figsize=(18, 5.5))

    for idx, (F, title) in enumerate([
        (F1, 'f_err1 (sensor 1)'),
        (F2, 'f_err2 (sensor 2)'),
        (F3, 'f_err3 (sensor 3)'),
    ]):
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
        plot_one(ax, X, Y, F, title)

    fig.tight_layout()
    plt.savefig('figure/f_err_eef_jacobian.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
