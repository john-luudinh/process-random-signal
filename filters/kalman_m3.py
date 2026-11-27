"""filters/kalman_m3.py - Bo loc Kalman mo rong trang thai 5 chieu cho EE6423 (M3).

Cach 1 (Chinh):
  z[k] -> Tien loc Notch nhan qua (f_c, 2*f_c) -> Kalman 5 trang thai:
  x_k = [i_k, d_k, v_k, v_{k-1}, v_{k-2}]^T
  - i_k: dong tai trung binh (AR(1) phi_2500 quanh i_nom o Mode 2, hoac Random Walk phi=1 o Mode 3)
  - d_k: troi nhiet bac 1 dan dong boi i_{k-1}^2: d_k = a_th * d_{k-1} + b_th * i_{k-1}^2 + w_{d,k}
  - [v_k, v_{k-1}, v_{k-2}]^T: nhieu mau nen AR(3) voi nhieu kich thich trang e_k

Cach 2 (Doi chieu):
  Goi bo loc Kalman 6 trang thai dao dong gon (filters/kalman_fixed.py) kem bu nhiet bac 1 M2.

Nguyen tac: Hoan toan KHONG import config.py (mu tham so that).
"""
import numpy as np
from scipy import signal, linalg
from filters import kalman_fixed


def causal_notch_prefilter(z, fs, fc, Q=12.0, n_init=25):
    """Loc chan dai (IIR Notch) nhan qua 1 chieu tai fc va 2*fc.

    Khoi tao trang thai bo loc bang lfilter_zi * mean(z[:n_init]) de giam gai qua do tai k=0.
    """
    z = np.asarray(z, dtype=float)
    z_init = float(np.mean(z[:min(n_init, len(z))]))

    b1, a1 = signal.iirnotch(fc, Q=Q, fs=fs)
    zi1 = signal.lfilter_zi(b1, a1) * z_init
    z1, _ = signal.lfilter(b1, a1, z, zi=zi1)

    b2, a2 = signal.iirnotch(2.0 * fc, Q=Q, fs=fs)
    zi2 = signal.lfilter_zi(b2, a2) * z_init
    z_out, _ = signal.lfilter(b2, a2, z1, zi=zi2)
    return z_out


def build_kalman_m3_matrices(phi, a_th, a_ar3, q_i, q_d, sigma_e2, R_adc):
    """Xay dung cac ma tran F (5x5), H (1x5), Q (5x5), R (1x1)."""
    a1, a2, a3 = [float(v) for v in a_ar3]
    F = np.array([
        [phi, 0.0,  0.0, 0.0, 0.0],
        [0.0, a_th, 0.0, 0.0, 0.0],
        [0.0, 0.0,  a1,  a2,  a3],
        [0.0, 0.0,  1.0, 0.0, 0.0],
        [0.0, 0.0,  0.0, 1.0, 0.0],
    ], dtype=float)

    H = np.array([[1.0, 1.0, 1.0, 0.0, 0.0]], dtype=float)
    Q = np.diag([float(q_i), float(q_d), float(sigma_e2), 0.0, 0.0])
    R = np.array([[float(R_adc)]], dtype=float)
    return F, H, Q, R


def check_observability_and_dare(F, H, Q, R):
    """Kiem tra hang ma tran kha quan sat O va giai phuong trinh Riccati xac lap (DARE)."""
    n = F.shape[0]
    O_blocks = [H @ np.linalg.matrix_power(F, k) for k in range(n)]
    O = np.vstack(O_blocks)
    rank_O = int(np.linalg.matrix_rank(O))
    cond_O = float(np.linalg.cond(O))

    # Giai DARE: P = F P F^T - F P H^T (H P H^T + R)^-1 H P F^T + Q
    P_dare = linalg.solve_discrete_are(F.T, H.T, Q, R)
    S_dare = float((H @ P_dare @ H.T + R)[0, 0])
    K_dare = (P_dare @ H.T)[:, 0] / S_dare
    return dict(
        O=O, rank_O=rank_O, cond_O=cond_O,
        P_dare=P_dare, S_dare=S_dare, K_dare=K_dare
    )


def run_kalman_m3(z_in, F, H, Q, R, b_th=0.0, i_nom=0.0, x0=None, P0=None):
    """Chay bo loc Kalman 5 trang thai voi cap nhat Joseph on dinh so.

    Phuong trinh du bao:
      x_{k|k-1} = F x_{k-1|k-1} + u_{k-1}
    trong do:
      u_{k-1}[0] = (1 - phi) * i_nom   (dua i_k ve quanh i_nom neu phi < 1)
      u_{k-1}[1] = b_th * (i_{k-1|k-1})^2  (tu phat nhiet bac 1)
    """
    z_in = np.asarray(z_in, dtype=float)
    n = len(z_in)
    nx = F.shape[0]
    phi = float(F[0, 0])

    if x0 is None:
        i0 = float(np.mean(z_in[:min(25, n)]))
        x = np.array([i0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    else:
        x = np.array(x0, dtype=float).copy()

    if P0 is None:
        P = np.diag([0.05, 1e-4, 1e-4, 1e-4, 1e-4])
    else:
        P = np.array(P0, dtype=float).copy()

    xs = np.empty((n, nx), dtype=float)
    P_diag = np.empty((n, nx), dtype=float)
    nu = np.empty(n, dtype=float)
    Sv = np.empty(n, dtype=float)
    nis = np.empty(n, dtype=float)

    I_nx = np.eye(nx)
    H0 = H[0]
    R_scalar = float(R[0, 0])
    K = np.zeros(nx, dtype=float)

    for k in range(n):
        # 1. Du bao (Predict)
        i_prev = x[0]
        x_pred = F @ x
        x_pred[0] += (1.0 - phi) * i_nom
        x_pred[1] += b_th * (i_prev ** 2)
        P_pred = F @ P @ F.T + Q

        # 2. Cap nhat (Update - dang Joseph)
        S = float((H @ P_pred @ H.T)[0, 0] + R_scalar)
        K = (P_pred @ H.T)[:, 0] / S
        e = float(z_in[k] - np.dot(H0, x_pred))
        x = x_pred + K * e

        IKH = I_nx - np.outer(K, H0)
        P = IKH @ P_pred @ IKH.T + np.outer(K, K) * R_scalar
        P = 0.5 * (P + P.T)

        xs[k] = x
        P_diag[k] = np.diag(P)
        nu[k] = e
        Sv[k] = S
        nis[k] = (e * e) / S

    return dict(
        xs=xs,
        i_hat=xs[:, 0],
        d_hat=xs[:, 1],
        v_hat=xs[:, 2],
        P_diag=P_diag,
        P_last=P,
        K_last=K,
        nu=nu,
        Sv=Sv,
        nis=nis,
    )


def run_kalman_method2_comparison(z_raw, fs, fc, params, q_di=200.0, q_r1=2e-6, q_r2=5e-7,
                                  R_m2=None, rho=0.9995, warm_start=False):
    """Cach 2 (Doi chieu): Kalman 6 trang thai dao dong gon (kalman_fixed.py) + bu nhiet M2."""
    z_raw = np.asarray(z_raw, dtype=float)
    if R_m2 is None:
        # Tong phuong sai nhieu nen (theo A^2)
        R_m2 = float(params["sigma_e2_A"] * 1.4)

    # Uoc luong bu nhiet bac 1 truc tuyen bang mo hinh M2 (a_th, b_th)
    a_th = params["a_th"]
    b_th = params["b_th"]
    F6, Q6, H6, R6 = kalman_fixed.build(fs, fc, q_di=q_di, q_r1=q_r1, q_r2=q_r2, R=R_m2, rho=rho)

    # Chay 1 luot uoc luong tho -> tinh nhiet -> bu nhiet, hoac cap nhat tuan tu tung mau
    n = len(z_raw)
    x = np.zeros(6)
    x[0] = float(np.mean(z_raw[:min(25, n)]))
    d_th = (params["g_th_V"] / params["S_I"]) * (x[0] ** 2) if warm_start else 0.0
    if warm_start:
        x[0] -= d_th

    P = np.eye(6) * 0.1
    I6 = np.eye(6)
    i_hat = np.empty(n, dtype=float)
    d_hat = np.empty(n, dtype=float)
    nu = np.empty(n, dtype=float)
    Sv = np.empty(n, dtype=float)

    for k in range(n):
        d_th = a_th * d_th + b_th * (x[0] ** 2)
        x = F6 @ x
        P = F6 @ P @ F6.T + Q6
        S = float((H6 @ P @ H6.T + R6)[0, 0])
        K = (P @ H6.T)[:, 0] / S
        e = float((z_raw[k] - d_th) - (H6 @ x)[0])
        x = x + K * e
        P = (I6 - np.outer(K, H6[0])) @ P
        P = 0.5 * (P + P.T)
        i_hat[k] = x[0]
        d_hat[k] = d_th
        nu[k] = e
        Sv[k] = S

    return dict(i_hat=i_hat, d_hat=d_hat, nu=nu, Sv=Sv, nis=(nu ** 2) / Sv)
