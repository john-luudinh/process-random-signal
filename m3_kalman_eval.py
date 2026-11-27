"""m3_kalman_eval.py - Danh gia toan dien Bo loc Kalman M3 (Che do 2 & Che do 3),
kiem dinh chuoi doi moi (NIS / Ljung-Box), phan tich do nhay Q/R va xuat 6 hinh bao cao.
"""
import os
import numpy as np
from scipy import signal, stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analysis_utils as U
import m2_analysis_calib as M2
import m3_params_from_m2 as P3
from filters import kalman_m3 as K3

plt.rcParams.update({"font.size": 9, "figure.dpi": 140})


def moving_avg(z, M):
    k = np.ones(M) / M
    return np.convolve(z, k, mode="full")[:len(z)]


def lpf1(z, fc, fs):
    a = np.exp(-2.0 * np.pi * fc / fs)
    return signal.lfilter([1.0 - a], [1.0, -a], z, zi=np.array([a * z[0]]))[0]


def step_rise_time_ms(t, sig, t_step=6.0, pre_win=(4.5, 5.9), post_win=(7.5, 9.5)):
    """Tinh thoi gian dap ung 10% -> 90% tai buoc nhay t = 6.0 s (Mode 3)."""
    m_pre = (t >= pre_win[0]) & (t <= pre_win[1])
    m_post = (t >= post_win[0]) & (t <= post_win[1])
    v0 = float(np.mean(sig[m_pre]))
    v1 = float(np.mean(sig[m_post]))
    dv = v1 - v0
    m_step = (t >= t_step) & (t <= t_step + 1.5)
    ts = t[m_step]
    ys = sig[m_step]
    # Vi dong giam khi tai tang (hoac tang khi tai tang: i tang tu ~1.6A -> ~3.3A)
    if dv >= 0:
        idx10 = np.where(ys >= v0 + 0.10 * dv)[0]
        idx90 = np.where(ys >= v0 + 0.90 * dv)[0]
    else:
        idx10 = np.where(ys <= v0 + 0.10 * dv)[0]
        idx90 = np.where(ys <= v0 + 0.90 * dv)[0]
    t10 = ts[idx10[0]] if len(idx10) else np.nan
    t90 = ts[idx90[0]] if len(idx90) else np.nan
    return float((t90 - t_step) * 1e3) if np.isfinite(t90) else np.nan, float((t90 - t10) * 1e3) if (np.isfinite(t10) and np.isfinite(t90)) else np.nan


def compute_metrics(est, x_true, skip=500):
    err = est[skip:] - x_true[skip:]
    rmse_mA = float(np.sqrt(np.mean(err ** 2)) * 1e3)
    mae_mA = float(np.mean(np.abs(err)) * 1e3)
    bias_mA = float(np.mean(err) * 1e3)
    tail_bias_mA = float(np.mean(est[-12500:] - x_true[-12500:]) * 1e3)  # 5 s cuoi
    return dict(rmse=rmse_mA, mae=mae_mA, bias=bias_mA, tail_bias=tail_bias_mA)


def plot_fig1_ar3_ljung_box(params):
    """Hinh 1: So sanh ACF phan du va Ljung-Box p-value cho AR(1), AR(2), AR(3), AR(4)."""
    ar_models = params["ar_models"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8))

    # Panel 1: ACF phan du AR(1), AR(2), AR(3)
    ax = axes[0]
    N_r = len(ar_models[3]["resid"])
    band = 1.96 / np.sqrt(N_r)
    lags = np.arange(1, 21)
    markers = {1: "s--", 2: "^-.", 3: "o-"}
    colors = {1: "C3", 2: "C1", 3: "C2"}
    for p in (1, 2, 3):
        rho = ar_models[p]["lb"]["rho"][1:21]
        ax.plot(lags, rho, markers[p], color=colors[p], ms=4, lw=1.1,
                label=f"AR({p}) (Q_LB={ar_models[p]['lb']['Q']:.1f}, p={ar_models[p]['lb']['p_value']:.4f})")
    ax.axhline(0, color="k", lw=0.6)
    ax.axhline(band, color="0.4", ls="--", lw=0.9, label=f"+-1.96/sqrt(N) = +-{band:.4f}")
    ax.axhline(-band, color="0.4", ls="--", lw=0.9)
    ax.set_xticks(lags)
    ax.set_title("M3.1a: ACF phan du nhieu nen I=0 theo bac AR(p)")
    ax.set_xlabel("Lag l")
    ax.set_ylabel("rho_hat[l]")
    ax.legend(fontsize=7.3)
    ax.grid(True, alpha=0.25)

    # Panel 2: Ljung-Box p-value va AIC theo bac p
    ax = axes[1]
    ps = [1, 2, 3, 4]
    pvals = [ar_models[p]["lb"]["p_value"] for p in ps]
    aics = [ar_models[p]["aic"] - ar_models[3]["aic"] for p in ps]
    bars = ax.bar(ps, pvals, color=["C3", "C1", "C2", "C0"], alpha=0.85, width=0.5)
    ax.axhline(0.05, color="r", ls="--", lw=1.1, label="Nguong alpha = 0.05 (dat nhieu trang)")
    for b, pv, da in zip(bars, pvals, aics):
        ax.text(b.get_x() + b.get_width() / 2, pv + 0.006,
                f"p={pv:.3f}\ndAIC={da:+.1f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(ps)
    ax.set_xticklabels([f"AR({p})" for p in ps])
    ax.set_ylim(0, max(pvals) + 0.05)
    ax.set_title("M3.1b: Kiem dinh Ljung-Box (m=20) & Delta AIC")
    ax.set_ylabel("p-value Ljung-Box")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    out = "figures/fig_m3_1_noise_ar3_lb.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> Da luu: {out}")


def plot_fig2_mode2(t2, z2, x2, res_m2, fs):
    """Hinh 2: Ket qua Kalman M3 tren Che do 2 (mien thoi gian + KTC 95% + PSD)."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), gridspec_kw={"width_ratios": [1.25, 1.0]})

    # Panel 1: Phong to 1.5 s (tu 10.0 s den 11.5 s)
    ax = axes[0]
    m = (t2 >= 10.0) & (t2 <= 11.5)
    i_hat = res_m2["i_hat"]
    sd_i = np.sqrt(res_m2["P_diag"][:, 0])
    ax.plot(t2[m], z2[m], color="0.78", lw=0.5, label="Do tho z[k] (co gon)")
    ax.fill_between(t2[m], i_hat[m] - 1.96 * sd_i[m], i_hat[m] + 1.96 * sd_i[m],
                    color="C0", alpha=0.22, label="KTC hau nghiem +-1.96*sigma_i")
    ax.plot(t2[m], i_hat[m], "C0", lw=1.2, label="Kalman M3 (5 trang thai)")
    ax.plot(t2[m], x2[m], "k--", lw=1.1, label="Ground truth i_avg[k]")
    ax.set_title("M3.2a: Bam dong tai OU o Che do 2 (phong to 10.0 - 11.5 s)")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("Dong dien (A)")
    ax.legend(fontsize=7.2, loc="upper right")
    ax.grid(True, alpha=0.25)

    # Panel 2: So sanh PSD truoc va sau loc
    ax = axes[1]
    fz, Pz = U.welch_psd(z2[500:], fs, 4096)
    fk, Pk = U.welch_psd(i_hat[500:], fs, 4096)
    fx, Px = U.welch_psd(x2[500:], fs, 4096)
    ax.semilogy(fz[1:], Pz[1:], color="0.7", lw=0.7, label="Do tho z[k]")
    ax.semilogy(fk[1:], Pk[1:], "C0", lw=1.1, label="Kalman M3 i_hat[k]")
    ax.semilogy(fx[1:], Px[1:], "k--", lw=1.0, label="Ground truth x[k]")
    ax.set_ylim(1e-10, 1e-1)
    ax.set_title("M3.2b: Pho PSD truoc va sau Loc Kalman (Che do 2)")
    ax.set_xlabel("f (Hz)")
    ax.set_ylabel("PSD (A^2/Hz)")
    ax.legend(fontsize=7.2)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    out = "figures/fig_m3_2_mode2_tracking.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> Da luu: {out}")


def plot_fig3_mode3(t3, z3, x3, d3_true_A, ma50, lpf20, res_m3):
    """Hinh 3: Che do 3 - Bam dong toan dai, phong to buoc nhay 6 s & uoc luong troi nhiet."""
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.2), gridspec_kw={"width_ratios": [1.4, 1.0]})

    # Panel (0,0): Toan dai 30 s
    ax = axes[0, 0]
    ax.plot(t3, z3, color="0.80", lw=0.35, label="Do tho z[k]")
    ax.plot(t3, ma50, "C1", lw=0.8, alpha=0.85, label="MA (M=50, 20 ms)")
    ax.plot(t3, res_m3["i_hat"], "C0", lw=1.1, label="Kalman M3 (co bu nhiet)")
    ax.plot(t3, x3, "k--", lw=0.9, label="Ground truth x[k]")
    ax.set_title("M3.3a: Che do 3 - Bam dong nhay bac & doc (0 - 30 s)")
    ax.set_ylabel("Dong dien (A)")
    ax.legend(fontsize=7.2, loc="upper left")
    ax.grid(True, alpha=0.25)

    # Panel (0,1): Phong to buoc nhay tai t = 6 s (5.92 -> 6.45 s)
    ax = axes[0, 1]
    m = (t3 >= 5.92) & (t3 <= 6.45)
    ax.plot(t3[m], z3[m], color="0.80", lw=0.5, label="Do tho")
    ax.plot(t3[m], ma50[m], "C1", lw=1.1, label="MA (20 ms)")
    ax.plot(t3[m], lpf20[m], "C2-.", lw=1.1, label="LPF (20 Hz)")
    ax.plot(t3[m], res_m3["i_hat"][m], "C0", lw=1.3, label="Kalman M3")
    ax.plot(t3[m], x3[m], "k--", lw=1.1, label="Ground truth")
    ax.set_title("M3.3b: Phong to dap ung qua do tai t = 6.0 s")
    ax.set_ylabel("Dong dien (A)")
    ax.legend(fontsize=7.0, loc="lower right")
    ax.grid(True, alpha=0.25)

    # Panel (1,0): Uoc luong troi nhiet d_hat[k] vs dV_th/S_I that
    ax = axes[1, 0]
    ax.plot(t3, d3_true_A * 1e3, "k--", lw=1.2, label="Troi nhiet that dV_th/S_I (doi chieu)")
    ax.plot(t3, res_m3["d_hat"] * 1e3, "C3", lw=1.3, label="Kalman uoc luong d_hat[k] (tu M2)")
    ax.set_title("M3.3c: Uoc luong & bu troi nhiet truc tuyen (Che do 3)")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("Troi nhiet quy doi (mA)")
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.25)

    # Panel (1,1): Sai so tuc thoi (i_hat - x) cua MA50 vs Kalman M3
    ax = axes[1, 1]
    err_ma = (ma50 - x3) * 1e3
    err_kf = (res_m3["i_hat"] - x3) * 1e3
    ax.plot(t3, err_ma, "C1", lw=0.6, alpha=0.7, label=f"MA50 (lech cuoi: {np.mean(err_ma[-12500:]):+.1f} mA)")
    ax.plot(t3, err_kf, "C0", lw=0.7, label=f"Kalman M3 (lech cuoi: {np.mean(err_kf[-12500:]):+.1f} mA)")
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_ylim(-60, 60)
    ax.set_title("M3.3d: Sai so uoc luong (i_hat - x) theo thoi gian")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("Sai so (mA)")
    ax.legend(fontsize=7.2)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    out = "figures/fig_m3_3_mode3_tracking_drift.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> Da luu: {out}")


def plot_fig4_innovation_nis(t2, res_m2, res_m3):
    """Hinh 4: Kiem dinh chuoi doi moi chuan hoa & NIS."""
    skip = 500
    nu2_norm = res_m2["nu"][skip:] / np.sqrt(res_m2["Sv"][skip:])
    nu3_norm = res_m3["nu"][skip:] / np.sqrt(res_m3["Sv"][skip:])
    lb2 = P3.ljung_box_test(nu2_norm, max_lag=20, fit_dof=0)
    lb3 = P3.ljung_box_test(nu3_norm, max_lag=20, fit_dof=0)

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.6))

    # Panel 1: Chuoi doi moi chuan hoa Mode 2
    ax = axes[0]
    t_sub = t2[skip:skip + 2500]  # 1 s mau
    ax.plot(t_sub, nu2_norm[:2500], "C0", lw=0.55)
    ax.axhline(1.96, color="C3", ls="--", lw=0.9, label="+-1.96")
    ax.axhline(-1.96, color="C3", ls="--", lw=0.9)
    in95 = float(np.mean(np.abs(nu2_norm) <= 1.96) * 100)
    ax.set_title(f"M3.4a: Doi moi chuan hoa Mode 2\n(trong +-1.96: {in95:.2f}%, NIS={np.mean(res_m2['nis'][skip:]):.3f})")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("nu_k / sqrt(S_k)")
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.25)

    # Panel 2: Histogram vs N(0, 1)
    ax = axes[1]
    bins = np.linspace(-4, 4, 60)
    ax.hist(nu2_norm, bins=bins, density=True, color="C0", alpha=0.65, label="Mode 2 (WSS)")
    xg = np.linspace(-4, 4, 200)
    ax.plot(xg, stats.norm.pdf(xg, 0, 1), "k--", lw=1.3, label="N(0, 1) ly thuyet")
    ax.set_title(f"M3.4b: Phan phoi doi moi Mode 2\n(mean={nu2_norm.mean():+.3f}, std={nu2_norm.std():.3f})")
    ax.set_xlabel("nu_k / sqrt(S_k)")
    ax.set_ylabel("Mat do")
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.25)

    # Panel 3: ACF doi moi Mode 2 vs Mode 3
    ax = axes[2]
    lags = np.arange(1, 21)
    band2 = 1.96 / np.sqrt(len(nu2_norm))
    ax.plot(lags, lb2["rho"][1:21], "o-", color="C0", ms=3.5, lw=1.0,
            label=f"Mode 2 (LB p={lb2['p_value']:.3f})")
    ax.plot(lags, lb3["rho"][1:21], "s--", color="C3", ms=3.5, lw=1.0,
            label=f"Mode 3 (LB p={lb3['p_value']:.4f})")
    ax.axhline(band2, color="0.4", ls="--", lw=0.8, label="+-1.96/sqrt(N)")
    ax.axhline(-band2, color="0.4", ls="--", lw=0.8)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(lags[::2])
    ax.set_title("M3.4c: ACF chuoi doi moi (Mode 2 vs Mode 3)")
    ax.set_xlabel("Lag l")
    ax.set_ylabel("rho_hat[l]")
    ax.legend(fontsize=7.2)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    out = "figures/fig_m3_4_innovation_nis.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> Da luu: {out}")
    return dict(lb2=lb2, lb3=lb3, in95_m2=in95)


def plot_fig5_sensitivity(z2_nf, x2, z3_nf, x3, params, q_i_m3_base):
    """Hinh 5: Phan tich do nhay sieu tham so alpha_Q (tren q_i) va alpha_R (tren R_adc)."""
    alphas_Q = np.logspace(-2, 2, 9)  # 0.01 .. 100
    rmse2_q, rmse3_q, nis2_q, nis3_q = [], [], [], []
    i_nom2 = float(np.mean(z2_nf[:500]))

    for aQ in alphas_Q:
        F2, H2, Q2, R2 = K3.build_kalman_m3_matrices(
            params["phi_2500"], params["a_th"], params["a_ar3"],
            params["q_i_mode2"] * aQ, params["q_d"], params["sigma_e2_A"], params["R_adc"]
        )
        d0_2 = (params["g_th_V"] / params["S_I"]) * (i_nom2 ** 2)
        r2 = K3.run_kalman_m3(z2_nf, F2, H2, Q2, R2, b_th=params["b_th"], i_nom=i_nom2 - d0_2,
                              x0=[i_nom2 - d0_2, d0_2, 0, 0, 0])
        rmse2_q.append(compute_metrics(r2["i_hat"], x2)["rmse"])
        nis2_q.append(float(np.mean(r2["nis"][500:])))

        F3, H3, Q3, R3 = K3.build_kalman_m3_matrices(
            1.0, params["a_th"], params["a_ar3"],
            q_i_m3_base * aQ, params["q_d"], params["sigma_e2_A"], params["R_adc"]
        )
        r3 = K3.run_kalman_m3(z3_nf, F3, H3, Q3, R3, b_th=params["b_th"], i_nom=0.0,
                              x0=[float(np.mean(z3_nf[:25])), 0.0, 0, 0, 0])
        rmse3_q.append(compute_metrics(r3["i_hat"], x3)["rmse"])
        nis3_q.append(float(np.mean(r3["nis"][500:])))

    alphas_R = np.array([0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0])
    rmse2_r, rmse3_r = [], []
    for aR in alphas_R:
        F2, H2, Q2, R2 = K3.build_kalman_m3_matrices(
            params["phi_2500"], params["a_th"], params["a_ar3"],
            params["q_i_mode2"], params["q_d"], params["sigma_e2_A"], params["R_adc"] * aR
        )
        d0_2 = (params["g_th_V"] / params["S_I"]) * (i_nom2 ** 2)
        r2 = K3.run_kalman_m3(z2_nf, F2, H2, Q2, R2, b_th=params["b_th"], i_nom=i_nom2 - d0_2,
                              x0=[i_nom2 - d0_2, d0_2, 0, 0, 0])
        rmse2_r.append(compute_metrics(r2["i_hat"], x2)["rmse"])

        F3, H3, Q3, R3 = K3.build_kalman_m3_matrices(
            1.0, params["a_th"], params["a_ar3"],
            q_i_m3_base, params["q_d"], params["sigma_e2_A"], params["R_adc"] * aR
        )
        r3 = K3.run_kalman_m3(z3_nf, F3, H3, Q3, R3, b_th=params["b_th"], i_nom=0.0,
                              x0=[float(np.mean(z3_nf[:25])), 0.0, 0, 0, 0])
        rmse3_r.append(compute_metrics(r3["i_hat"], x3)["rmse"])

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.6))

    ax = axes[0]
    ax.semilogx(alphas_Q, rmse2_q, "o-", color="C0", label="Mode 2 (WSS)")
    ax.semilogx(alphas_Q, rmse3_q, "s-", color="C3", label="Mode 3 (Nhay bac)")
    ax.axvline(1.0, color="k", ls="--", lw=0.9, label="Gia tri danh dinh (alpha=1)")
    ax.set_title("M3.5a: Do nhay RMSE theo he so alpha_Q")
    ax.set_xlabel("alpha_Q (he so nhan q_i)")
    ax.set_ylabel("RMSE (mA)")
    ax.legend(fontsize=7.3)
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    ax.loglog(alphas_Q, nis2_q, "o-", color="C0", label="NIS Mode 2")
    ax.loglog(alphas_Q, nis3_q, "s-", color="C3", label="NIS Mode 3")
    ax.axhline(1.0, color="k", ls="--", lw=0.9, label="Ly thuyet E[NIS] = 1")
    ax.set_title("M3.5b: Trung binh NIS theo alpha_Q")
    ax.set_xlabel("alpha_Q (he so nhan q_i)")
    ax.set_ylabel("Mean NIS")
    ax.legend(fontsize=7.3)
    ax.grid(True, alpha=0.25)

    ax = axes[2]
    ax.semilogx(alphas_R, rmse2_r, "o-", color="C0", label="Mode 2 (WSS)")
    ax.semilogx(alphas_R, rmse3_r, "s-", color="C3", label="Mode 3 (Nhay bac)")
    ax.axvline(1.0, color="k", ls="--", lw=0.9, label="R_adc danh dinh (alpha=1)")
    ax.set_title("M3.5c: Do nhay RMSE theo he so alpha_R")
    ax.set_xlabel("alpha_R (he so nhan R_adc)")
    ax.set_ylabel("RMSE (mA)")
    ax.legend(fontsize=7.3)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    out = "figures/fig_m3_5_sensitivity_qr.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> Da luu: {out}")
    return dict(alphas_Q=alphas_Q, rmse2_q=rmse2_q, rmse3_q=rmse3_q, nis2_q=nis2_q, nis3_q=nis3_q)


def plot_fig6_mode3_stft(t3, z3, z3_nf, res_m3, fs, fc_hat):
    """Hinh 6: Han che cua Notch co dinh o Che do 3 (STFT dich tan so gon co gop)."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))

    # Panel 1: STFT cua z3 tho quanh dai 250 - 500 Hz
    ax = axes[0]
    f_st, t_st, Zxx = signal.stft(z3 - moving_avg(z3, 125), fs=fs, nperseg=1024, noverlap=768)
    m_f = (f_st >= 280) & (f_st <= 460)
    pcm = ax.pcolormesh(t_st, f_st[m_f], 10 * np.log10(np.abs(Zxx[m_f]) ** 2 + 1e-12),
                        shading="gouraud", cmap="inferno", vmin=-55, vmax=-10)
    ax.axhline(fc_hat, color="cyan", ls="--", lw=1.2, label=f"Notch co dinh f_c = {fc_hat:.1f} Hz")
    fig.colorbar(pcm, ax=ax, label="Cong suat (dB)")
    ax.set_title("M3.6a: STFT cua z[k] Mode 3 (f_c dich theo tai)")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("Tan so f (Hz)")
    ax.legend(fontsize=7.5, loc="lower left")

    # Panel 2: STFT sau Notch co dinh (phan gon lot qua khi f_c dich)
    ax = axes[1]
    f_st2, t_st2, Zxx2 = signal.stft(res_m3["i_hat"] - moving_avg(res_m3["i_hat"], 125),
                                     fs=fs, nperseg=1024, noverlap=768)
    m_f2 = (f_st2 >= 280) & (f_st2 <= 460)
    pcm2 = ax.pcolormesh(t_st2, f_st2[m_f2], 10 * np.log10(np.abs(Zxx2[m_f2]) ** 2 + 1e-12),
                         shading="gouraud", cmap="inferno", vmin=-55, vmax=-10)
    ax.axhline(fc_hat, color="cyan", ls="--", lw=1.2, label=f"Notch co dinh f_c = {fc_hat:.1f} Hz")
    fig.colorbar(pcm2, ax=ax, label="Cong suat (dB)")
    ax.set_title("M3.6b: STFT sau Notch co dinh + Kalman (gon lot o t<6s & t>26s)")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("Tan so f (Hz)")
    ax.legend(fontsize=7.5, loc="lower left")

    fig.tight_layout()
    out = "figures/fig_m3_6_mode3_ripple_mismatch.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> Da luu: {out}")


def main():
    os.makedirs("figures", exist_ok=True)
    params = P3.estimate_m3_params(verbose=True)
    data = M2.load_data()
    fs = params["fs"]
    VQ, SI = params["V_Q"], params["S_I"]

    # Kiem tra ma tran he thong & kha quan sat
    F2, H2, Q2, R2 = K3.build_kalman_m3_matrices(
        params["phi_2500"], params["a_th"], params["a_ar3"],
        params["q_i_mode2"], params["q_d"], params["sigma_e2_A"], params["R_adc"]
    )
    obs2 = K3.check_observability_and_dare(F2, H2, Q2, R2)

    # Mode 3: phi = 1 (Random walk cho dong tai nhay bac + doc), q_i lon hon de bam buoc nhay nhanh
    # Chon sigma_wi = 6 mA/mau (q_i = 3.6e-5 A^2) tu phan tich danh doi toc do dap ung
    q_i_m3 = (6.0e-3) ** 2
    F3, H3, Q3, R3 = K3.build_kalman_m3_matrices(
        1.0, params["a_th"], params["a_ar3"],
        q_i_m3, params["q_d"], params["sigma_e2_A"], params["R_adc"]
    )
    obs3 = K3.check_observability_and_dare(F3, H3, Q3, R3)

    print("=" * 76)
    print("M3 - KIEM TRA KHA QUAN SAT & NGHIEM RICCATI (DARE)")
    print("=" * 76)
    print(f"  Mode 2 (phi={params['phi_2500']:.5f}): rank(O) = {obs2['rank_O']}/5 | "
          f"K_dare = {np.round(obs2['K_dare'], 4)} | sqrt(S_dare) = {np.sqrt(obs2['S_dare'])*1e3:.3f} mA")
    print(f"  Mode 3 (phi=1.00000): rank(O) = {obs3['rank_O']}/5 | "
          f"K_dare = {np.round(obs3['K_dare'], 4)} | sqrt(S_dare) = {np.sqrt(obs3['S_dare'])*1e3:.3f} mA")
    assert obs2["rank_O"] == 5 and obs3["rank_O"] == 5, "He thong mat tinh quan sat duoc!"

    # ---------------- CHE DO 2 ----------------
    d2 = data[2]
    z2 = (d2["y"] - VQ) / SI
    x2 = d2["x"]
    z2_nf = K3.causal_notch_prefilter(z2, fs, params["fc_hat"], Q=12.0)

    i_nom2 = float(np.mean(z2_nf[:500]))
    d0_2 = (params["g_th_V"] / SI) * (i_nom2 ** 2)
    i0_2 = i_nom2 - d0_2

    # Kalman M3 (khong bu nhiet vs co bu nhiet)
    F2_nobu, H2_nobu, Q2_nobu, R2_nobu = K3.build_kalman_m3_matrices(
        params["phi_2500"], 1.0, params["a_ar3"],
        params["q_i_mode2"], 0.0, params["sigma_e2_A"], params["R_adc"]
    )
    res_m2_nobu = K3.run_kalman_m3(z2_nf, F2_nobu, H2_nobu, Q2_nobu, R2_nobu,
                                   b_th=0.0, i_nom=i_nom2, x0=[i_nom2, 0.0, 0, 0, 0])
    res_m2 = K3.run_kalman_m3(z2_nf, F2, H2, Q2, R2,
                              b_th=params["b_th"], i_nom=i0_2, x0=[i0_2, d0_2, 0, 0, 0])
    res_m2_c2 = K3.run_kalman_method2_comparison(z2, fs, params["fc_hat"], params, warm_start=True)

    # ---------------- CHE DO 3 ----------------
    d3 = data[3]
    z3 = (d3["y"] - VQ) / SI
    x3 = d3["x"]
    d3_true_A = d3["dV_th"] / SI
    z3_nf = K3.causal_notch_prefilter(z3, fs, params["fc_hat"], Q=12.0)

    i0_3 = float(np.mean(z3_nf[:25]))
    res_m3_nobu = K3.run_kalman_m3(z3_nf, F3, H3, Q3, R3, b_th=0.0, i_nom=0.0, x0=[i0_3, 0.0, 0, 0, 0])
    res_m3 = K3.run_kalman_m3(z3_nf, F3, H3, Q3, R3, b_th=params["b_th"], i_nom=0.0, x0=[i0_3, 0.0, 0, 0, 0])
    res_m3_c2 = K3.run_kalman_method2_comparison(z3, fs, params["fc_hat"], params, warm_start=False)

    # Baseline
    M_per = int(round(fs / params["fc_hat"]))
    ma7_2, ma7_3 = moving_avg(z2, M_per), moving_avg(z3, M_per)
    ma50_2, ma50_3 = moving_avg(z2, 50), moving_avg(z3, 50)
    lpf20_2, lpf20_3 = lpf1(z2, 20.0, fs), lpf1(z3, 20.0, fs)

    methods = [
        ("Raw z[k]", z2, z3),
        ("Notch nhan qua", z2_nf, z3_nf),
        (f"MA (M={M_per}, ~1 T_c)", ma7_2, ma7_3),
        ("MA (M=50, 20 ms)", ma50_2, ma50_3),
        ("LPF bac 1 (20 Hz)", lpf20_2, lpf20_3),
        ("Kalman M3 (khong bu nhiet)", res_m2_nobu["i_hat"], res_m3_nobu["i_hat"]),
        ("Kalman M3 (Cach 1, co bu nhiet)", res_m2["i_hat"], res_m3["i_hat"]),
        ("Kalman 6TT (Cach 2, co bu nhiet)", res_m2_c2["i_hat"], res_m3_c2["i_hat"]),
    ]

    print("\n" + "=" * 92)
    print("BANG DOI CHIEU HIEU NANG UOC LUONG DONG (CHE DO 2 & CHE DO 3)")
    print("=" * 92)
    print(f"{'Phuong phap':33s} | {'M2 RMSE':>8s} | {'M2 Bias':>8s} | {'M3 RMSE':>8s} | {'M3 Bias5s':>9s} | {'M3 t_90(ms)':>11s}")
    print("-" * 92)
    for name, s2, s3 in methods:
        m2 = compute_metrics(s2, x2)
        m3 = compute_metrics(s3, x3)
        t90, _ = step_rise_time_ms(d3["t"], s3)
        print(f"{name:33s} | {m2['rmse']:8.2f} | {m2['bias']:+8.2f} | {m3['rmse']:8.2f} | {m3['tail_bias']:+9.2f} | {t90:11.2f}")

    # Kiem tra do phu KTC 95% hau nghiem o Mode 2
    cov95_m2 = float(np.mean(np.abs(res_m2["i_hat"][500:] - x2[500:]) <= 1.96 * np.sqrt(res_m2["P_diag"][500:, 0])) * 100)
    print(f"\n  [Kiem tra P_k] Mode 2 do phu KTC 95% hau nghiem: {cov95_m2:.2f}%")
    print(f"  [Nhiet Mode 3] Tai t=30s: d_hat = {res_m3['d_hat'][-1]*1e3:.2f} mA vs d_true = {d3_true_A[-1]*1e3:.2f} mA")

    # Xuat 6 hinh & kiem dinh doi moi
    print("\n" + "=" * 76)
    print("XUAT 6 BIEU DO M3 & KIEM DINH DOI MOI")
    print("=" * 76)
    plot_fig1_ar3_ljung_box(params)
    plot_fig2_mode2(d2["t"], z2, x2, res_m2, fs)
    plot_fig3_mode3(d3["t"], z3, x3, d3_true_A, ma50_3, lpf20_3, res_m3)
    inn = plot_fig4_innovation_nis(d2["t"], res_m2, res_m3)
    sens = plot_fig5_sensitivity(z2_nf, x2, z3_nf, x3, params, q_i_m3)
    plot_fig6_mode3_stft(d3["t"], z3, z3_nf, res_m3, fs, params["fc_hat"])

    print(f"\n  [Sensitivity Q] alpha_Q=0.01 -> M2 RMSE={sens['rmse2_q'][0]:.2f} mA, M3 RMSE={sens['rmse3_q'][0]:.2f} mA")
    print(f"  [Sensitivity Q] alpha_Q=1.00 -> M2 RMSE={sens['rmse2_q'][4]:.2f} mA, M3 RMSE={sens['rmse3_q'][4]:.2f} mA")
    print(f"  [Sensitivity Q] alpha_Q=100  -> M2 RMSE={sens['rmse2_q'][-1]:.2f} mA, M3 RMSE={sens['rmse3_q'][-1]:.2f} mA")

    nu2_norm = res_m2["nu"][500:] / np.sqrt(res_m2["Sv"][500:])
    nu3_norm = res_m3["nu"][500:] / np.sqrt(res_m3["Sv"][500:])
    print(f"\n  [Innovation Mode 2] mean={nu2_norm.mean():+.4f}, std={nu2_norm.std():.4f}, "
          f"in +-1.96={inn['in95_m2']:.2f}%, mean(NIS)={np.mean(res_m2['nis'][500:]):.3f}, "
          f"LB(20) Q={inn['lb2']['Q']:.2f}, p={inn['lb2']['p_value']:.4f}")
    print(f"  [Innovation Mode 3] mean={nu3_norm.mean():+.4f}, std={nu3_norm.std():.4f}, "
          f"mean(NIS)={np.mean(res_m3['nis'][500:]):.3f}, "
          f"LB(20) Q={inn['lb3']['Q']:.2f}, p={inn['lb3']['p_value']:.4f}")


if __name__ == "__main__":
    main()
