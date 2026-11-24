"""m2_quicklook.py - Kiem tra nhanh mo hinh: hieu chuan (che do 1), thong ke/ACF/Welch (che do 2),
baseline RMSE (che do 2, 3). Sinh figures/fig_*.png va in bang so lieu."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C
import sim_dc_acs712 as S
import analysis_utils as U

plt.rcParams.update({"font.size": 9, "figure.dpi": 130})
fs = C.F_S

# ---------- Che do 1: hieu chuan LS  y = V_Q + S_I * I_ref ----------
d1 = S.run_mode(1)
seg = int(C.MODE1["T_each"] * fs)
skip = int(0.05 * fs)                                   # bo 50 ms qua do cua loc
I_all, y_all, means = [], [], []
for k, I in enumerate(C.MODE1["I_ref"]):
    yy = d1["y"][k * seg + skip:(k + 1) * seg]
    I_all.append(np.full(len(yy), I)); y_all.append(yy); means.append(U.mean_ci(yy))
I_all = np.concatenate(I_all); y_all = np.concatenate(y_all)
H = np.column_stack([np.ones_like(I_all), I_all])
th, cov, r = U.ls_fit(H, y_all)
kap = np.mean([m["kappa"] for m in means])
se = np.sqrt(np.diag(cov) * kap)                       # hieu chinh tri nho
V_Q, S_I = th
print(f"[M1 calib] V_Q = {V_Q*1e3:.2f} mV +- {1.96*se[0]*1e3:.2f} (that {1e3*(2.5+C.SENSOR['dV0']):.2f})")
print(f"[M1 calib] S_I = {S_I*1e3:.3f} mV/A +- {1.96*se[1]*1e3:.3f} (that {C.SENSOR['S_true']*1e3:.3f}); kappa~{kap:.2f}")
print(f"[M1 calib] sigma tai I=0: {means[0]['std']*1e3:.2f} mV ; kappa={means[0]['kappa']:.2f}, Neff={means[0]['Neff']:.0f}")

def to_amp(y):
    return (y - V_Q) / S_I

# ---------- Che do 2: thong ke, ACF, Welch ----------
d2 = S.run_mode(2)
z2 = to_amp(d2["y"]); x2 = d2["x"]
m2 = U.mean_ci(z2)
print(f"[M2 stats] mean z = {m2['mean']:.4f} A +- {m2['ci']:.4f} (ngay tho +-{m2['half_naive']:.4f}), "
      f"kappa={m2['kappa']:.1f}, Neff={m2['Neff']:.0f}; mean x = {x2.mean():.4f}")
r, rho = U.acf_biased(z2, 400)
w_mean = d2["w"].mean(); f_rot = w_mean / 2 / np.pi; f_c = C.MOTOR["N_seg"] * f_rot
print(f"[M2] f_rot = {f_rot:.1f} Hz, f_c = {f_c:.1f} Hz (w={2*np.pi*f_c/fs:.3f} rad/sample), 2f_c = {2*f_c:.1f} Hz")
alias = abs(C.EMI['f_smps'] - round(C.EMI['f_smps'] / fs) * fs)
print(f"[M2] SMPS alias -> {alias:.0f} Hz (w = {2*np.pi*alias/fs/np.pi:.3f} pi)")

fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
for L, c in [(256, "C0"), (2048, "C1"), (8192, "C2")]:
    f, P = U.welch_psd(z2, fs, L)
    ax[0].semilogy(f, P, c, lw=0.8, label=f"Welch L={L}")
# thi nghiem bo Sallen-Key -> thay alias SMPS
C.FILTER["use_sallen_key"] = False
d2n = S.run_mode(2); C.FILTER["use_sallen_key"] = True
f, P = U.welch_psd(to_amp(d2n["y"]), fs, 2048)
ax[0].semilogy(f, P, "0.6", lw=0.7, label="khong Sallen-Key, L=2048")
for fr, lab in [(f_c, "f_c"), (2 * f_c, "2f_c"), (alias, "alias SMPS")]:
    ax[0].axvline(fr, color="k", ls=":", lw=0.7); ax[0].text(fr, ax[0].get_ylim()[1] * 0.3, lab, fontsize=7)
ax[0].set_xlabel("f (Hz)  [w = 2 pi f / 2500]"); ax[0].set_ylabel("PSD (A^2/Hz)"); ax[0].legend(fontsize=7)
ax[0].set_title("Che do 2: PSD dong do (da tru trung binh)")
lags = np.arange(len(rho)) / fs * 1e3
ax[1].plot(lags, rho, lw=0.8); ax[1].axhline(0, color="k", lw=0.5)
ax[1].set_xlabel("lag (ms)"); ax[1].set_ylabel("rho_hat[l]"); ax[1].set_title("ACF chuan hoa (chia N)")
fig.tight_layout(); fig.savefig("figures/fig_m2_psd_acf.png")

# ---------- Baseline ----------
def moving_avg(z, M):
    k = np.ones(M) / M
    return np.convolve(z, k, mode="full")[:len(z)]          # nhan qua
def lpf1(z, fc):
    a = np.exp(-2 * np.pi * fc / fs); y = np.empty_like(z); s = z[0]
    for n, v in enumerate(z):
        s = a * s + (1 - a) * v; y[n] = s
    return y

d3 = S.run_mode(3)
z3 = to_amp(d3["y"]); x3 = d3["x"]
M_per = int(round(fs / f_c))                              # ~1 chu ky gon
rows = []
for name, fn in [("Raw", lambda z: z), (f"MA M={M_per}", lambda z: moving_avg(z, M_per)),
                 ("MA M=50 (20 ms)", lambda z: moving_avg(z, 50)), ("LPF bac1 fc=20 Hz", lambda z: lpf1(z, 20))]:
    rows.append((name, U.rmse(fn(z2)[500:], x2[500:]), U.rmse(fn(z3)[500:], x3[500:])))
print("[Baseline RMSE (A)]  ten | che do 2 | che do 3")
for r_ in rows:
    print(f"   {r_[0]:22s} {r_[1]:.4f}   {r_[2]:.4f}")

fig, ax = plt.subplots(2, 1, figsize=(9, 4.4), sharex=True)
ax[0].plot(d3["t"], z3, color="0.75", lw=0.3, label="so do da hieu chuan z[n]")
ax[0].plot(d3["t"], moving_avg(z3, 50), "C1", lw=0.8, label="MA 20 ms")
ax[0].plot(d3["t"], x3, "k", lw=1, label="ground truth i_avg")
ax[0].set_ylabel("A"); ax[0].legend(fontsize=7, loc="upper left"); ax[0].set_title("Che do 3: tai nhay bac / doc + tu phat nhiet")
ax[1].plot(d3["t"], d3["dV_th"] / S_I * 1e3, "C3"); ax[1].set_ylabel("troi nhiet (mA tuong duong)")
ax[1].set_xlabel("t (s)")
fig.tight_layout(); fig.savefig("figures/fig_m3_timeseries.png")

# ---------- Yule-Walker cho phan du sau khi bo gon (minh hoa) ----------
a, s2 = U.yule_walker(z2 - z2.mean(), 4)
print("[YW AR(4)] a =", np.round(a, 3), " sigma_e =", np.sqrt(s2))
print("[drift] dV_th cuoi che do 3 = %.2f mV ~ %.1f mA" % (d3['dV_th'][-1]*1e3, d3['dV_th'][-1]/S_I*1e3))
