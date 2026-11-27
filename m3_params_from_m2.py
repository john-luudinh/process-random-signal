"""m3_params_from_m2.py - Trich xuat & nang cap tham so tu M2 phuc vu M3 (Kalman).

Nguyen tac "Mu tham so that":
- Khong import tham so vat ly an tu config.py (chi doc thong so datasheet ADC: bits, Vref).
- Moi tham so dong hoc (V_Q, S_I, AR(3), phi_load, tau_th, kP*TC, f_c) deu uoc luong
  truc tiep tu du lieu do data/mode1.npz va data/mode2.npz.
"""
import contextlib
import io
import numpy as np
from scipy import signal, stats

import config as C
import analysis_utils as U
import m2_analysis_calib as M2


def ljung_box_test(residuals, max_lag=20, fit_dof=0):
    """Kiem dinh Ljung-Box Q_LB(m) cho chuoi phan du.

    Tra ve dict(Q, df, p_value, max_lag, rho).
    """
    r = np.asarray(residuals, dtype=float)
    r = r - r.mean()
    N = len(r)
    _, rho = U.acf_biased(r, max_lag)
    lags = np.arange(1, max_lag + 1)
    Q = float(N * (N + 2) * np.sum((rho[1:max_lag + 1] ** 2) / (N - lags)))
    df = max(max_lag - fit_dof, 1)
    p_val = float(1.0 - stats.chi2.cdf(Q, df))
    return dict(Q=Q, df=df, p_value=p_val, max_lag=max_lag, rho=rho)


def fit_ar_p(x, p):
    """Uoc luong mo hinh AR(p) bang binh phuong toi thieu (LS) va tinh AIC/BIC + Ljung-Box."""
    x = np.asarray(x, dtype=float)
    N_eff = len(x) - p
    H = np.column_stack([x[p - k - 1:len(x) - k - 1] for k in range(p)])
    y = x[p:]
    theta, cov, resid = U.ls_fit(H, y)
    se = np.sqrt(np.diag(cov))
    sigma_e2 = float(np.var(resid, ddof=p))
    aic = float(N_eff * np.log(sigma_e2) + 2 * p)
    bic = float(N_eff * np.log(sigma_e2) + p * np.log(N_eff))
    lb = ljung_box_test(resid, max_lag=20, fit_dof=p)
    return dict(
        p=p, a=theta, se=se, resid=resid,
        sigma_e2=sigma_e2, sigma_e=float(np.sqrt(sigma_e2)),
        aic=aic, bic=bic, lb=lb
    )


def estimate_m3_params(verbose=True):
    """Chay pipeline uoc luong tham so tu M2 va nang cap cho M3.

    Tra ve dict day du tham so cho bo loc Kalman M3 va cac bang kiem dinh.
    """
    data = M2.load_data()
    fs = float(data[1]["fs"])

    # Tat stdout cua ham M2 neu khong can in lai toan bo bang D6/D7
    with contextlib.redirect_stdout(io.StringIO()):
        calib = M2.d6_calibration(data)
        d7 = M2.d7_calibrated_model_check(data, calib)

    VQ = calib["V_Q"]
    SI = calib["S_I"]

    # 1. Phan tich nhieu nen tai I=0 (10 s dau Mode 1, bo 50 ms qua do)
    segs = M2._extract_mode1_segments(data[1], skip_s=0.050)
    v_V = segs[0]["y"] - segs[0]["y"].mean()
    v_A = v_V / SI

    ar_models = {p: fit_ar_p(v_A, p) for p in (1, 2, 3, 4)}
    ar3 = ar_models[3]

    # 2. Uoc luong dong tai AR(1) o toc do ha mau 100 Hz (tranh chech do nhieu nen 2500 Hz)
    #    va quy doi sang fs = 2500 Hz
    z2 = (data[2]["y"] - VQ) / SI
    z2_zm = z2 - z2.mean()
    fc_hat = float(d7["rip1"]["fc"])
    b1, a1 = signal.iirnotch(fc_hat, Q=12.0, fs=fs)
    b2, a2 = signal.iirnotch(2.0 * fc_hat, Q=12.0, fs=fs)
    z2_res = signal.filtfilt(b2, a2, signal.filtfilt(b1, a1, z2_zm))

    decim_load = 25  # 2500 Hz -> 100 Hz
    fs_load = fs / decim_load
    z2_100 = signal.decimate(z2_res, decim_load, zero_phase=True)
    ar1_100 = fit_ar_p(z2_100, 1)
    phi_100 = float(ar1_100["a"][0])
    se_phi_100 = float(ar1_100["se"][0])
    phi_2500 = float(phi_100 ** (fs_load / fs))
    f_load_100 = float(-fs_load * np.log(max(phi_100, 1e-6)) / (2.0 * np.pi))

    # Phuong sai dong tai dai thap tu M2 (var_x_est) -> suy ra phuong sai nhieu qua trinh tai
    var_x = float(d7["var_x_est"])
    q_i_mode2 = float(var_x * (1.0 - phi_2500 ** 2))

    # 3. Tham so nhiet bac 1 tu M2 (quy doi tu V sang A)
    g_th_V = float(d7["g_th"])          # kP * TC [V / A^2]
    tau_th = float(d7["tau_th"])        # [s]
    a_th = float(np.exp(-1.0 / (fs * tau_th)))
    b_th = float((g_th_V / SI) * (1.0 - a_th))  # [A / A^2] moi buoc Ts

    # Phuong sai qua trinh nhiet (cho phep Kalman bam sai lech mo hinh nhiet nhe)
    # Tu do bat dinh toc do nhiet o M2: se_rate [V/(A^2.s)] -> quy doi sang buoc Ts tai dong ~2.6 A
    se_rate_A = float(d7["se_rate"] / SI)
    sigma_wd_step = max(se_rate_A * (2.6 ** 2) / fs, 1e-6)
    q_d = float(sigma_wd_step ** 2)

    # 4. Nhieu luong tu hoa ADC tu datasheet (bits, Vref) quy doi sang A^2
    lsb_V = float(C.ADC["Vref"] / (2 ** C.ADC["bits"]))
    delta_A = lsb_V / SI
    R_adc = float((delta_A ** 2) / 12.0)

    params = dict(
        fs=fs,
        V_Q=VQ, ci_VQ=calib["ci_VQ"],
        S_I=SI, ci_SI=calib["ci_SI"],
        kappa_bar=calib["kappa_bar"],
        fc_hat=fc_hat,
        fc2_hat=float(d7["rip2"]["fc"]),
        rip1=d7["rip1"],
        rip2=d7["rip2"],
        ar_models=ar_models,
        a_ar3=ar3["a"],
        sigma_e_A=ar3["sigma_e"],
        sigma_e2_A=ar3["sigma_e2"],
        phi_100=phi_100,
        se_phi_100=se_phi_100,
        phi_2500=phi_2500,
        f_load_100=f_load_100,
        var_x_est=var_x,
        q_i_mode2=q_i_mode2,
        g_th_V=g_th_V,
        tau_th=tau_th,
        a_th=a_th,
        b_th=b_th,
        q_d=q_d,
        lsb_V=lsb_V,
        delta_A=delta_A,
        R_adc=R_adc,
    )

    if verbose:
        print("=" * 76)
        print("M3 - THAM SO HIEU CHUAN TU M2 & NANG CAP PHAN HOI M1-M2")
        print("=" * 76)
        print(f"  [Hieu chuan tinh M2] V_Q = {VQ*1e3:.3f} +- {calib['ci_VQ']*1e3:.3f} mV | "
              f"S_I = {SI*1e3:.3f} +- {calib['ci_SI']*1e3:.3f} mV/A")
        print(f"  [Gon co gop M2]      f_c = {fc_hat:.3f} Hz | 2f_c = {params['fc2_hat']:.3f} Hz")
        print("\n  [So sanh AR(1)..AR(4) nhieu nen I=0 & Kiem dinh Ljung-Box (m=20)]")
        print("   Bac p | He so a_hat                            | sigma_e (mA) |   AIC     | Q_LB(20) | p-value | Trang?")
        for p in (1, 2, 3, 4):
            m = ar_models[p]
            a_str = ", ".join(f"{v:+.4f}" for v in m["a"])
            lb = m["lb"]
            is_white = "DAT" if lb["p_value"] > 0.05 else "KHONG"
            print(f"     {p}   | [{a_str:38s}] |    {m['sigma_e']*1e3:6.3f}    | {m['aic']:9.1f} |  {lb['Q']:6.2f}  | {lb['p_value']:7.4f} | {is_white}")
        print("\n  [Dong tai AR(1) ha mau 100 Hz -> quy doi 2500 Hz]")
        print(f"    phi_100 = {phi_100:.4f} +- {1.96*se_phi_100:.4f} (f_load = {f_load_100:.3f} Hz, "
              f"LB p={ar1_100['lb']['p_value']:.4f})")
        print(f"    phi_2500 = {phi_2500:.5f} | var_x = {var_x:.4e} A^2 | q_i_mode2 = {q_i_mode2:.4e} A^2")
        print("\n  [Nhiet bac 1 & ADC]")
        print(f"    kP*TC = {g_th_V*1e3:.3f} mV/A^2, tau_th = {tau_th:.1f} s -> "
              f"a_th = {a_th:.7f}, b_th = {b_th:.3e} A^-1, q_d = {q_d:.3e} A^2")
        print(f"    ADC 12-bit (5V): LSB = {lsb_V*1e3:.4f} mV ({delta_A*1e3:.3f} mA) -> "
              f"R_adc = {R_adc:.4e} A^2 (sigma_ADC = {np.sqrt(R_adc)*1e3:.3f} mA)\n")

    return params


if __name__ == "__main__":
    estimate_m3_params(verbose=True)
