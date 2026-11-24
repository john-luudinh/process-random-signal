"""m1_expected_psd.py - PSD DU KIEN (ly thuyet, khong uoc luong) cua z[n] o che do 2, tu cong thuc B1-B9.
Ve tren cung truc voi Welch cua du lieu mo phong de doi chieu (doi chieu dung cho M2).
PSD mot phia theo Hz (A^2/Hz) de so truc tiep voi scipy.signal.welch."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C

fs, Ts = C.F_S, 1 / C.F_S
m, s, flt = C.MOTOR, C.SENSOR, C.FILTER
S_I = s["S_true"]


def H_analog(f):
    s_ = 2j * np.pi * f
    w = 2 * np.pi * flt["f_sk"]
    return 1 / (1 + s_ * flt["RF"] * flt["CF"]) * w * w / (s_ * s_ + s_ * w / flt["Q_sk"] + w * w)


def H_tau_i(f):  # B1
    s_ = 2j * np.pi * f
    return m["Ke"] / ((m["La"] * s_ + m["Ra"]) * (m["J"] * s_ + m["B"]) + m["Ke"] * m["Kt"])


def S_x(f):  # B1: OU qua H_tau->i (mot phia)
    sig, tau = C.LOAD_NOISE["sigma"], C.LOAD_NOISE["tau"]
    S_tau = 4 * sig ** 2 * tau / (1 + (2 * np.pi * f * tau) ** 2)
    return np.abs(H_tau_i(f)) ** 2 * S_tau


def ripple(f, mu, var_x, f_c):  # B3-B4: buou Gauss tai f_c, 2f_c
    sig_w = m["Ra"] / m["Ke"] * np.sqrt(var_x)
    out = np.zeros_like(f)
    for k, g in [(1, m["gamma1"]), (2, m["gamma2"])]:
        gt = g * abs(H_analog(k * f_c))
        P = gt ** 2 / 2 * (mu ** 2 + var_x)
        sd = k * m["N_seg"] * sig_w / (2 * np.pi)          # Hz (hai bac 2 khuech tan gap doi)
        out += P * np.exp(-(f - k * f_c) ** 2 / (2 * sd ** 2)) / (np.sqrt(2 * np.pi) * sd)
    return out


def S_c(f, kmax=40):  # B7: Hall trang qua loc, cong ban sao (mot phia -> theo A^2/Hz)
    fc_ds = 1 / (2 * np.pi * flt["RF"] * 47e-9)
    N0 = s["sigma_hall_2k"] ** 2 / (np.pi / 2 * fc_ds)
    tot = np.zeros_like(f)
    for k in range(-kmax, kmax + 1):
        tot += N0 * np.abs(H_analog(f - k * fs)) ** 2
    return tot / S_I ** 2


def S_v(f):  # B8
    lsb = C.ADC["Vref"] / 2 ** C.ADC["bits"]
    var = lsb ** 2 / 12 + (C.ADC["sigma_adc_lsb"] * lsb) ** 2
    return np.full_like(f, 2 * var / fs) / S_I ** 2


if __name__ == "__main__":
    import sim_dc_acs712 as S, analysis_utils as U
    f = np.linspace(0.05, fs / 2, 20000)
    w_ss = S.steady_state(C.MODE2["tauL0"]); mu = w_ss[0]
    f_c = m["N_seg"] * w_ss[1] / (2 * np.pi)
    var_x = np.trapezoid(S_x(f), f)
    parts = {"x (tai)": S_x(f), "gon": ripple(f, mu, var_x, f_c), "Hall": S_c(f), "luong tu": S_v(f)}
    total = sum(parts.values())
    print(f"mu={mu:.3f} A, sigma_x={np.sqrt(var_x):.4f} A, f_c={f_c:.1f} Hz")
    print("cong suat tung phan (A^2):", {k: f"{np.trapezoid(v, f):.3e}" for k, v in parts.items()})
    d = S.run_mode(2)
    z = (d["y"] - (0.5 * s["Vcc"] + s["dV0"])) / S_I
    fw, Pw = U.welch_psd(z, fs, 8192)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.semilogy(fw, Pw, color="0.7", lw=0.7, label="Welch L=8192 (uoc luong, M2)")
    for k, v in parts.items():
        ax.semilogy(f, v, lw=0.8, ls="--", label=f"ly thuyet: {k}")
    ax.semilogy(f, total, "k", lw=1, label="ly thuyet: tong")
    ax.set_ylim(1e-9, 1e-1); ax.set_xlabel("f (Hz);  w = 2 pi f / 2500"); ax.set_ylabel("A^2/Hz")
    ax.legend(fontsize=7, ncol=2); ax.set_title("Che do 2: PSD du kien (M1) va Welch (M2)")
    fig.tight_layout(); fig.savefig("figures/fig_m1_expected_psd.png", dpi=130)
    # so sanh so: tren cac dai
    for a, b in [(0.5, 5), (5, 50), (100, 300), (360, 390), (900, 1200)]:
        mk, mw = (f >= a) & (f < b), (fw >= a) & (fw < b)
        print(f"dai {a}-{b} Hz: ly thuyet {np.trapezoid(total[mk], f[mk]):.2e}  Welch {np.trapezoid(Pw[mw], fw[mw]):.2e} A^2")
