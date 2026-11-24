"""kalman_fixed.py - Kalman 6 trang thai (tu lap trinh), tan so gon CO DINH.
x = [i_avg, di, r1a, r1b, r2a, r2b];  z[n] = i_avg + r1a + r2a + v[n]  (z da hieu chuan sang A)
 - i_avg: mo hinh tich phan buoc ngau nhien (toc do bien thien la random walk)
 - (r1a,r1b): bo dao dong quay goc Om1 = 2*pi*f_c/fs moi mau, he so tat rho
 - (r2a,r2b): hai bac 2, goc 2*Om1
"""
import numpy as np


def rot(theta, rho):
    c, s = np.cos(theta), np.sin(theta)
    return rho * np.array([[c, -s], [s, c]])


def build(fs, f_c, q_di, q_r1, q_r2, R, rho=0.9995):
    T = 1.0 / fs
    Om = 2 * np.pi * f_c / fs
    F = np.zeros((6, 6))
    F[:2, :2] = [[1, T], [0, 1]]
    F[2:4, 2:4] = rot(Om, rho)
    F[4:6, 4:6] = rot(2 * Om, rho)
    Q = np.diag([0.0, q_di, q_r1, q_r1, q_r2, q_r2])
    H = np.array([[1.0, 0, 1, 0, 1, 0]])
    return F, Q, H, np.array([[R]])


def run(z, F, Q, H, R, x0=None, P0=None):
    n = len(z)
    x = np.zeros(6) if x0 is None else x0.copy()
    if x0 is None:
        x[0] = z[:20].mean()
    P = np.eye(6) if P0 is None else P0.copy()
    xs = np.empty((n, 6)); nu = np.empty(n); Sv = np.empty(n)
    I6 = np.eye(6)
    for k in range(n):
        # du bao
        x = F @ x
        P = F @ P @ F.T + Q
        # cap nhat
        S = (H @ P @ H.T + R)[0, 0]
        K = (P @ H.T)[:, 0] / S
        e = z[k] - (H @ x)[0]
        x = x + K * e
        P = (I6 - np.outer(K, H[0])) @ P
        P = 0.5 * (P + P.T)
        xs[k] = x; nu[k] = e; Sv[k] = S
    return xs, nu, Sv
