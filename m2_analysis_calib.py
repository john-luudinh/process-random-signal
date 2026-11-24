"""m2_analysis_calib.py - Phan tich du lieu va hieu chuan cam bien (EE6423, D2 - D6).

Nguyen tac:
- Moi con so uoc luong tu mot ban ghi di kem sai so tinh tu chinh ban ghi do.
- Gia tri that trong config.py CHI dung de doi chieu/cham diem, KHONG dung de tinh.
- Moi buoc (D2, D3, D4, D5, D6) la mot ham doc lap va xuat mot hinh trong figures/.
"""
import os
import numpy as np
from scipy import signal, optimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C
import analysis_utils as U
import m1_expected_psd as M1
import sim_dc_acs712 as S_sim

plt.rcParams.update({"font.size": 9, "figure.dpi": 140})


_trapz = getattr(np, "trapezoid", np.trapz)


def load_data():
    """Doc du lieu 3 che do tu data/mode{1,2,3}.npz (sinh boi sim_dc_acs712.py)."""
    data = {}
    for m in (1, 2, 3):
        path = f"data/mode{m}.npz"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Chua co {path}. Hay chay 'python sim_dc_acs712.py' truoc.")
        with np.load(path) as f:
            data[m] = {k: f[k] for k in f.files}
            data[m]["fs"] = float(data[m]["fs"])
    return data


def _extract_mode1_segments(d1, skip_s=0.050):
    """Tach 17 doan cua Che do 1 theo C.MODE1['seq'], bo skip_s (50 ms) dau moi doan."""
    fs = d1["fs"]
    skip = int(round(skip_s * fs))
    y1, t1 = d1["y"], d1["t"]
    segs = []
    idx0 = 0
    for k, (I_val, T_seg) in enumerate(C.MODE1["seq"]):
        n_seg = int(round(T_seg * fs))
        idx1 = idx0 + n_seg
        s_start = idx0 + skip
        yy = y1[s_start:idx1]
        tt = t1[s_start:idx1]
        segs.append(dict(
            k=k, I_ref=float(I_val), T_seg=float(T_seg),
            idx0=idx0, idx1=idx1, s_start=s_start,
            t=tt, t_mid=float(tt.mean()), y=yy,
            stats=U.mean_ci(yy)
        ))
        idx0 = idx1
    return segs


# ==============================================================================
# D2 - KIEM TRA THO DU LIEU & QUA DO CHUYEN MUC
# ==============================================================================
def d2_raw_check(data):
    """D2 - Ve ma ADC va dong danh dinh z_nom = (y - 2.5)/0.185 cho 3 che do,
    kiem tra bao hoa ADC (0/4095) va thoi gian xac lap buoc nhay o Che do 1 (< 5 ms).
    """
    print("=" * 76)
    print("D2 - KIEM TRA THO DU LIEU & QUA DO CHUYEN MUC")
    print("=" * 76)

    for m in (1, 2, 3):
        code = data[m]["code"]
        y = data[m]["y"]
        z_nom = (y - 2.5) / 0.185
        n_sat = int(np.sum((code == 0) | (code == 4095)))
        m_y = U.mean_ci(y)
        m_z = U.mean_ci(z_nom)
        print(f"  [Che do {m}] N = {len(code):6d} mau ({len(code)/data[m]['fs']:.1f} s) | "
              f"ADC code: min={code.min():4d}, max={code.max():4d}, bao hoa={n_sat}")
        print(f"             y_mean = {m_y['mean']:.4f} +- {m_y['ci']:.4f} V | "
              f"z_nom_mean = {m_z['mean']:.4f} +- {m_z['ci']:.4f} A (kappa={m_z['kappa']:.1f})")
        assert n_sat == 0, f"Che do {m} bi bao hoa ADC!"

    d1 = data[1]
    fs = d1["fs"]
    y1, t1 = d1["y"], d1["t"]
    z1_nom = (y1 - 2.5) / 0.185
    seq = C.MODE1["seq"]
    skip_50ms = int(0.050 * fs)

    idx0 = 0
    t_settle_list = []
    worst_step_info = None
    max_dz = -1.0

    for k, (I_val, T_seg) in enumerate(seq):
        n_seg = int(round(T_seg * fs))
        idx1 = idx0 + n_seg
        seg_z = z1_nom[idx0:idx1]
        ss_part = seg_z[skip_50ms:]
        mu_ss = ss_part.mean()
        std_ss = ss_part.std()

        if k > 0:
            head = seg_z[:skip_50ms]
            dz_step = abs(mu_ss - prev_mu_ss)
            thresh = max(3.0 * std_ss, 0.02 * dz_step)
            outside = np.where(np.abs(head[:int(0.015 * fs)] - mu_ss) > thresh)[0]
            t_settle_ms = (outside[-1] + 1) / fs * 1e3 if len(outside) > 0 else 0.0
            t_settle_list.append(t_settle_ms)

            if dz_step > max_dz:
                max_dz = dz_step
                worst_step_info = dict(
                    k=k, t_edge=idx0 / fs, idx0=idx0, mu_ss=mu_ss, std_ss=std_ss,
                    prev_mu=prev_mu_ss, I_prev=seq[k - 1][0], I_new=I_val,
                    t_settle_ms=t_settle_ms
                )
        prev_mu_ss = mu_ss
        idx0 = idx1

    max_settle = max(t_settle_list)
    print(f"  [Qua do Che do 1] So buoc chuyen muc: {len(t_settle_list)} | "
          f"Thoi gian xac lap max: {max_settle:.2f} ms (< 5.0 ms: {max_settle < 5.0})")
    print(f"                    Bo 50 ms dau moi doan ({skip_50ms} mau) la du an toan.")

    os.makedirs("figures", exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.2))
    titles = {
        1: "Che do 1: Hieu chuan tinh (xen ke I_ref & doan I=0)",
        2: "Che do 2: Dong co tai on dinh (WSS)",
        3: "Che do 3: Tai nhay bac / doc + tu phat nhiet",
    }
    positions = {1: axes[0, 0], 2: axes[0, 1], 3: axes[1, 0]}

    for m in (1, 2, 3):
        ax = positions[m]
        tm, codem, ym = data[m]["t"], data[m]["code"], data[m]["y"]
        zm_nom = (ym - 2.5) / 0.185
        ax.plot(tm, codem, color="C0", lw=0.45)
        ax.axhline(0, color="r", ls="--", lw=0.7, alpha=0.6)
        ax.axhline(4095, color="r", ls="--", lw=0.7, alpha=0.6)
        ax.set_title(titles[m])
        ax.set_xlabel("t (s)")
        ax.set_ylabel("Ma ADC (12-bit)", color="C0")
        ax.set_ylim(-150, 4250)
        ax.grid(True, alpha=0.25)

        lsb = 5.0 / 4096.0
        ax_r = ax.twinx()
        ax_r.set_ylim((-150 * lsb - 2.5) / 0.185, (4250 * lsb - 2.5) / 0.185)
        ax_r.set_ylabel("z_nom = (y - 2.5)/0.185 (A)", color="0.3")
        ax.text(0.02, 0.86, f"code in [{codem.min()}, {codem.max()}]\n"
                            f"z_nom in [{zm_nom.min():.2f}, {zm_nom.max():.2f}] A",
                transform=ax.transAxes, fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.9))

    ax_z = axes[1, 1]
    ws = worst_step_info
    sl = slice(ws["idx0"] - int(0.010 * fs), ws["idx0"] + int(0.060 * fs))
    t_rel_ms = (t1[sl] - ws["t_edge"]) * 1e3
    ax_z.plot(t_rel_ms, z1_nom[sl], "C0.-", lw=0.8, ms=2.5, label="z_nom[n]")
    ax_z.axhline(ws["mu_ss"], color="k", lw=0.8, label=f"TB xac lap ({ws['mu_ss']:.3f} A)")
    ax_z.axhspan(ws["mu_ss"] - 3 * ws["std_ss"], ws["mu_ss"] + 3 * ws["std_ss"],
                 color="C2", alpha=0.2, label="Dai +-3 sigma")
    ax_z.axvline(5.0, color="C3", ls="--", lw=1.0, label="Moc 5 ms")
    ax_z.axvline(50.0, color="C1", ls="-.", lw=1.1, label="Moc bo qua 50 ms")
    ax_z.axvspan(0.0, 50.0, color="C1", alpha=0.08)
    ax_z.set_title(f"Che do 1: Qua do buoc nhay {ws['I_prev']:.1f} -> {ws['I_new']:.1f} A (t={ws['t_edge']:.0f} s)")
    ax_z.set_xlabel("Thoi gian tu luc chuyen muc (ms)")
    ax_z.set_ylabel("z_nom (A)")
    ax_z.legend(fontsize=7.2, loc="lower right")
    ax_z.grid(True, alpha=0.25)

    fig.tight_layout()
    out_path = "figures/fig_d2_raw_check.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  -> Da luu hinh: {out_path}\n")
    return dict(max_settle_ms=max_settle, out_path=out_path)


# ==============================================================================
# D6 - HIEU CHUAN V_Q, S_I BANG LS (KHONG DUNG THAM SO CONFIG DE TINH)
# ==============================================================================
def d6_calibration(data):
    """D6 - Hieu chuan LS ngay tho vs LS co hieu chinh troi nhiet tu cac doan I=0.
    Tra ve dict(V_Q, S_I, ci_VQ, ci_SI, ...) de dung quy doi A cho D3, D4, D5.
    """
    print("=" * 76)
    print("D6 - HIEU CHUAN V_Q, S_I BANG BINH PHUONG TOI THIEU (LS)")
    print("=" * 76)

    d1 = data[1]
    segs = _extract_mode1_segments(d1, skip_s=0.050)
    kap_mean = float(np.mean([s["stats"]["kappa"] for s in segs]))

    # 1. LS ngay tho tren tat ca cac doan da bo 50 ms dau
    I_all = np.concatenate([np.full(len(s["y"]), s["I_ref"]) for s in segs])
    y_all = np.concatenate([s["y"] for s in segs])
    t_all = np.concatenate([s["t"] for s in segs])
    H_all = np.column_stack([np.ones_like(I_all), I_all])

    th_naive, cov_naive, r_naive = U.ls_fit(H_all, y_all)
    se_naive = np.sqrt(np.diag(cov_naive) * kap_mean)
    VQ_n, SI_n = th_naive

    # 2. LS hieu chinh troi nhiet tu 9 doan I=0 (k = 0, 2, 4, ..., 16)
    zero_segs = [s for s in segs if s["I_ref"] == 0.0]
    t0_pts = np.array([s["t_mid"] for s in zero_segs])
    mu0_pts = np.array([s["stats"]["mean"] for s in zero_segs])
    ci0_pts = np.array([s["stats"]["ci"] for s in zero_segs])

    # Noi suy tuyen tinh duong nen b_hat(t) va troi tuong doi so voi moc nguoi ban dau
    b_hat_all = np.interp(t_all, t0_pts, mu0_pts)
    drift_hat_all = b_hat_all - mu0_pts[0]
    y_corr_all = y_all - drift_hat_all

    # Fit LS tren cac doan I_ref > 0 (va ca toan bo de kiem chung)
    nz_segs = [s for s in segs if s["I_ref"] > 0.0]
    I_nz = np.concatenate([np.full(len(s["y"]), s["I_ref"]) for s in nz_segs])
    t_nz = np.concatenate([s["t"] for s in nz_segs])
    y_nz = np.concatenate([s["y"] for s in nz_segs])
    drift_nz = np.interp(t_nz, t0_pts, mu0_pts) - mu0_pts[0]
    y_nz_corr = y_nz - drift_nz
    H_nz = np.column_stack([np.ones_like(I_nz), I_nz])

    # Cach A: tru drift_hat truoc roi LS tren toan bo 17 doan (hoac 8 doan I>0)
    th_corr, cov_corr, r_corr = U.ls_fit(H_all, y_corr_all)
    se_corr = np.sqrt(np.diag(cov_corr) * kap_mean)
    # Cong them phuong sai cua moc goc mu0_pts[0] cho sai so cua V_Q
    se_corr_VQ = float(np.sqrt(se_corr[0] ** 2 + (ci0_pts[0] / 1.96) ** 2))
    se_corr_SI = float(se_corr[1])
    VQ_c, SI_c = th_corr

    # Cach B (kiem tra cheo): them cot drift_hat vao ma tran H -> cho ket qua tuong duong
    H_aug = np.column_stack([np.ones_like(I_all), I_all, drift_hat_all])
    th_aug, cov_aug, _ = U.ls_fit(H_aug, y_all)
    se_aug = np.sqrt(np.diag(cov_aug) * kap_mean)

    # Cham diem voi gia tri that trong config.py (CHI de doi chieu)
    VQ_true = 0.5 * C.SENSOR["Vcc"] + C.SENSOR["dV0"]
    SI_true = C.SENSOR["S_true"]

    in_VQ_n = abs(VQ_n - VQ_true) <= 1.96 * se_naive[0]
    in_SI_n = abs(SI_n - SI_true) <= 1.96 * se_naive[1]
    in_VQ_c = abs(VQ_c - VQ_true) <= 1.96 * se_corr_VQ
    in_SI_c = abs(SI_c - SI_true) <= 1.96 * se_corr_SI

    print(f"  He so tri nho trung binh cac doan: kappa_bar = {kap_mean:.2f}")
    print(f"  [LS ngay tho]      V_Q = {VQ_n*1e3:.3f} +- {1.96*se_naive[0]*1e3:.3f} mV  (chua that {VQ_true*1e3:.2f}? {in_VQ_n})")
    print(f"                     S_I = {SI_n*1e3:.3f} +- {1.96*se_naive[1]*1e3:.3f} mV/A (chua that {SI_true*1e3:.3f}? {in_SI_n})")
    print(f"  [LS hieu chinh]    V_Q = {VQ_c*1e3:.3f} +- {1.96*se_corr_VQ*1e3:.3f} mV  (chua that {VQ_true*1e3:.2f}? {in_VQ_c})")
    print(f"                     S_I = {SI_c*1e3:.3f} +- {1.96*se_corr_SI*1e3:.3f} mV/A (chua that {SI_true*1e3:.3f}? {in_SI_c})")
    print(f"  [LS cot H bo sung] V_Q = {th_aug[0]*1e3:.3f} +- {1.96*se_aug[0]*1e3:.3f} mV, "
          f"S_I = {th_aug[1]*1e3:.3f} +- {1.96*se_aug[1]*1e3:.3f} mV/A, alpha_drift = {th_aug[2]:.3f}")

    # 3. Kiem tra phan du tung muc va xu huong theo thoi gian
    res_n_means, res_n_cis = [], []
    res_c_means, res_c_cis = [], []
    seg_I, seg_tmid = [], []
    idx0 = 0
    for s in segs:
        n_s = len(s["y"])
        rn = r_naive[idx0:idx0 + n_s]
        rc = r_corr[idx0:idx0 + n_s]
        mn = U.mean_ci(rn)
        mc = U.mean_ci(rc)
        res_n_means.append(mn["mean"]); res_n_cis.append(mn["ci"])
        res_c_means.append(mc["mean"]); res_c_cis.append(mc["ci"])
        seg_I.append(s["I_ref"]); seg_tmid.append(s["t_mid"])
        idx0 += n_s

    # Kiem tra xu huong theo thoi gian cua phan du
    Ht = np.column_stack([np.ones_like(t_all), t_all])
    th_tn, cov_tn, _ = U.ls_fit(Ht, r_naive)
    th_tc, cov_tc, _ = U.ls_fit(Ht, r_corr)
    se_tn = np.sqrt(np.diag(cov_tn) * kap_mean)[1]
    se_tc = np.sqrt(np.diag(cov_tc) * kap_mean)[1]

    print(f"  [Xu huong phan du theo t] LS ngay tho  : slope = {th_tn[1]*1e3:+.4f} +- {1.96*se_tn*1e3:.4f} mV/s")
    print(f"                            LS hieu chinh: slope = {th_tc[1]*1e3:+.4f} +- {1.96*se_tc*1e3:.4f} mV/s")
    n_in_ci_n = sum(abs(m) <= c for m, c in zip(res_n_means, res_n_cis))
    n_in_ci_c = sum(abs(m) <= c for m, c in zip(res_c_means, res_c_cis))
    print(f"  [Phan du tung doan chua 0] Ngay tho: {n_in_ci_n}/{len(segs)} doan | Hieu chinh: {n_in_ci_c}/{len(segs)} doan")

    # 4. Ve hinh D6
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4))

    # Panel 1: Duong troi nhiet uoc luong tu cac doan I=0 vs dV_th that
    ax = axes[0, 0]
    ax.errorbar(t0_pts, (mu0_pts - mu0_pts[0]) * 1e3, yerr=ci0_pts * 1e3,
                fmt="o-", color="C0", capsize=3, lw=1.0, ms=4, label="Uoc luong b_hat(t) - b_hat(0) (+-KTC 95%)")
    ax.plot(d1["t"], d1["dV_th"] * 1e3, "k--", lw=1.0, label="dV_th that (config, de doi chieu)")
    ax.set_title("D6.1: Duong troi nhiet uoc luong tu cac doan I=0")
    ax.set_xlabel("t (s)"); ax.set_ylabel("Troi nhiet (mV)")
    ax.legend(fontsize=7.5); ax.grid(True, alpha=0.25)

    # Panel 2: Duong hoi quy LS hieu chinh
    ax = axes[0, 1]
    seg_yc_means = [np.mean(s["y"] - (np.interp(s["t"], t0_pts, mu0_pts) - mu0_pts[0])) for s in segs]
    I_grid = np.linspace(0, 4.1, 50)
    ax.scatter(seg_I, np.array(seg_yc_means), color="C0", s=28, zorder=3, label="TB tung doan (da tru troi)")
    ax.plot(I_grid, VQ_c + SI_c * I_grid, "C1-", lw=1.3,
            label=f"LS: V_Q={VQ_c*1e3:.2f} mV, S_I={SI_c*1e3:.3f} mV/A")
    ax.set_title("D6.2: Duong hieu chuan LS sau hieu chinh troi")
    ax.set_xlabel("I_ref (A)"); ax.set_ylabel("y_corr (V)")
    ax.legend(fontsize=7.5); ax.grid(True, alpha=0.25)

    # Panel 3: Phan du theo muc dong I_ref (chi ve 8 muc >0 va muc 0 dau)
    ax = axes[1, 0]
    ax.errorbar(np.array(seg_I) - 0.04, np.array(res_n_means) * 1e3, yerr=np.array(res_n_cis) * 1e3,
                fmt="s", color="C3", ms=4, capsize=3, lw=0.9, label="LS ngay tho (+-KTC 95%)")
    ax.errorbar(np.array(seg_I) + 0.04, np.array(res_c_means) * 1e3, yerr=np.array(res_c_cis) * 1e3,
                fmt="o", color="C2", ms=4, capsize=3, lw=0.9, label="LS hieu chinh troi (+-KTC 95%)")
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_title("D6.3: Trung binh phan du theo muc dong I_ref")
    ax.set_xlabel("I_ref (A)"); ax.set_ylabel("Phan du (mV)")
    ax.legend(fontsize=7.5); ax.grid(True, alpha=0.25)

    # Panel 4: Phan du theo thoi gian t
    ax = axes[1, 1]
    ax.errorbar(seg_tmid, np.array(res_n_means) * 1e3, yerr=np.array(res_n_cis) * 1e3,
                fmt="s-", color="C3", ms=3.5, capsize=2.5, lw=0.8, alpha=0.85,
                label=f"LS ngay tho (slope={th_tn[1]*1e3:+.3f} mV/s)")
    ax.errorbar(seg_tmid, np.array(res_c_means) * 1e3, yerr=np.array(res_c_cis) * 1e3,
                fmt="o-", color="C2", ms=3.5, capsize=2.5, lw=0.8,
                label=f"LS hieu chinh (slope={th_tc[1]*1e3:+.3f} mV/s)")
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_title("D6.4: Phan du theo thoi gian t (kiem tra con troi)")
    ax.set_xlabel("t (s)"); ax.set_ylabel("Phan du (mV)")
    ax.legend(fontsize=7.5); ax.grid(True, alpha=0.25)

    fig.tight_layout()
    out_path = "figures/fig_d6_calibration.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  -> Da luu hinh: {out_path}\n")

    return dict(
        V_Q=float(VQ_c), S_I=float(SI_c),
        ci_VQ=float(1.96 * se_corr_VQ), ci_SI=float(1.96 * se_corr_SI),
        V_Q_naive=float(VQ_n), S_I_naive=float(SI_n),
        ci_VQ_naive=float(1.96 * se_naive[0]), ci_SI_naive=float(1.96 * se_naive[1]),
        kappa_bar=kap_mean
    )


# ==============================================================================
# D3 - THONG KE CO BAN KEM KTC & KIEM TRA WSS
# ==============================================================================
def d3_basic_stats(data, calib):
    """D3 - Thong ke co ban kem KTC (chia N vs N-1, hieu chinh tri nho kappa,
    cach tan so cho Che do 2, kiem tra WSS nua dau vs nua sau).
    """
    print("=" * 76)
    print("D3 - THONG KE CO BAN KEM KTC & KIEM TRA WSS")
    print("=" * 76)
    VQ, SI = calib["V_Q"], calib["S_I"]
    to_amp = lambda y: (y - VQ) / SI

    # 1-3. Thong ke 17 doan Che do 1 (theo don vi A sau hieu chuan D6)
    segs = _extract_mode1_segments(data[1], skip_s=0.050)
    print("  [Che do 1 - cac doan sau 50 ms dau (quy doi A bang D6)]")
    print("   k | I_ref |    N  |  mu_hat  | s_N (mA) | s_N-1(mA)| kappa | N_eff | +-CI_corr | +-CI_naive")
    kappas_m1 = []
    for s in segs:
        z_seg = to_amp(s["y"])
        N = len(z_seg)
        mc = U.mean_ci(z_seg)
        s_N = float(np.sqrt(np.mean((z_seg - mc["mean"]) ** 2)))
        s_Nm1 = float(np.sqrt(np.sum((z_seg - mc["mean"]) ** 2) / (N - 1)))
        kappas_m1.append(mc["kappa"])
        print(f"  {s['k']:2d} |  {s['I_ref']:4.1f} | {N:5d} | {mc['mean']:8.4f} | {s_N*1e3:8.3f} | "
              f"{s_Nm1*1e3:8.3f} | {mc['kappa']:5.2f} | {mc['Neff']:5.0f} | {mc['ci']:9.5f} | {mc['half_naive']:9.5f}")

    # 4. Che do 2: quy tac lag am dau tien vs cach theo tan so Var[mu] ~ P(0)*fs/(2N)
    d2 = data[2]
    fs = d2["fs"]
    z2 = to_amp(d2["y"])
    N2 = len(z2)
    m2_acf = U.mean_ci(z2)
    s2_N = float(np.sqrt(np.mean((z2 - m2_acf["mean"]) ** 2)))
    s2_Nm1 = float(np.sqrt(np.sum((z2 - m2_acf["mean"]) ** 2) / (N2 - 1)))

    # Tinh Welch PSD cho Che do 2 de lay P_z(0) quanh f ~ 0
    fw, Pw = U.welch_psd(z2, fs, 2048)
    # Lay gia tri PSD o dai tan thap (bin dau tien > 0 den <= 2 Hz de tranh anh huong DC subtraction cua Hann)
    low_mask = (fw > 0) & (fw <= 2.5)
    P0 = float(np.mean(Pw[low_mask]))
    var_mu_freq = P0 * fs / (2.0 * N2)
    se_mu_freq = float(np.sqrt(var_mu_freq))
    ci_freq = 1.96 * se_mu_freq
    kappa_freq = float((P0 * fs / 2.0) / (s2_N ** 2))
    Neff_freq = N2 / kappa_freq

    x2_mean = float(d2["x"].mean())
    print("\n  [Che do 2 - So sanh 3 cach tinh KTC 95% cho mu_hat]")
    print(f"    N = {N2}, mu_hat = {m2_acf['mean']:.5f} A, sigma_N = {s2_N:.5f} A, sigma_N-1 = {s2_Nm1:.5f} A (lech {(s2_Nm1/s2_N - 1)*100:.4f}%)")
    print(f"    1) Ngay tho (gia su iid):       +- {m2_acf['half_naive']:.5f} A  (N_eff = {N2})")
    print(f"    2) ACF dung o lag<=0 (L={m2_acf['L']}):    +- {m2_acf['ci']:.5f} A  (kappa = {m2_acf['kappa']:.2f}, N_eff = {m2_acf['Neff']:.0f}) -> SAI vi gon!")
    print(f"    3) Tan so P(0)*fs/(2N):         +- {ci_freq:.5f} A  (P(0)={P0:.4e} A^2/Hz, kappa_freq={kappa_freq:.1f}, N_eff={Neff_freq:.0f})")
    print(f"    Doi chieu ground truth x2.mean() = {x2_mean:.5f} A | Nam trong KTC tan so? {abs(m2_acf['mean'] - x2_mean) <= ci_freq}")

    # 5. Kiem tra WSS cho Che do 2 (nua dau vs nua sau) va doi chung Che do 3
    def check_wss(z_arr, label):
        half = len(z_arr) // 2
        z_a, z_b = z_arr[:half], z_arr[half:]
        _, Pa = U.welch_psd(z_a, fs, 2048)
        _, Pb = U.welch_psd(z_b, fs, 2048)
        fa, _ = U.welch_psd(z_a, fs, 2048)
        m_mask = (fa > 0) & (fa <= 2.5)
        var_a = float(np.mean(Pa[m_mask])) * fs / (2.0 * len(z_a))
        var_b = float(np.mean(Pb[m_mask])) * fs / (2.0 * len(z_b))
        d_mu = float(z_a.mean() - z_b.mean())
        ci_diff = 1.96 * float(np.sqrt(var_a + var_b))
        v_a, v_b = float(z_a.var()), float(z_b.var())
        print(f"    [{label}] mu1={z_a.mean():.4f}, mu2={z_b.mean():.4f}, "
              f"delta_mu={d_mu:+.5f} +- {ci_diff:.5f} A (chua 0? {abs(d_mu) <= ci_diff}) | "
              f"var1={v_a:.5f}, var2={v_b:.5f}")
        return dict(mu1=float(z_a.mean()), mu2=float(z_b.mean()), d_mu=d_mu, ci_diff=ci_diff, var1=v_a, var2=v_b)

    print("\n  [Kiem tra WSS: nua dau vs nua sau]")
    wss_m2 = check_wss(z2, "Che do 2 (on dinh)")
    z3 = to_amp(data[3]["y"])
    wss_m3 = check_wss(z3, "Che do 3 (doi chung khong dung)")

    # Ve hinh D3
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))

    # Panel 1: kappa cac doan Che do 1
    ax = axes[0]
    ks = [s["k"] for s in segs]
    ax.bar(ks, kappas_m1, color=["C0" if s["I_ref"] == 0 else "C1" for s in segs], alpha=0.85)
    ax.axhline(1.0, color="k", ls="--", lw=0.8, label="kappa = 1 (nhieu trang)")
    ax.set_title("D3.1: He so tri nho kappa (Che do 1)")
    ax.set_xlabel("Chi so doan k (xanh: I=0, cam: I>0)")
    ax.set_ylabel("kappa_hat")
    ax.legend(fontsize=7.5); ax.grid(True, alpha=0.25)

    # Panel 2: So sanh 3 KTC cua mu_hat o Che do 2
    ax = axes[1]
    labels = ["Ngay tho\n(iid)", "ACF lag<=0\n(L=%d)" % m2_acf["L"], "Mien tan so\nP(0)*fs/(2N)"]
    cis = [m2_acf["half_naive"], m2_acf["ci"], ci_freq]
    colors = ["C3", "C1", "C2"]
    for i, (lab, ci_v, col) in enumerate(zip(labels, cis, colors)):
        ax.errorbar([i], [m2_acf["mean"]], yerr=[ci_v], fmt="o", color=col, capsize=5, lw=1.8, ms=6)
    ax.axhline(x2_mean, color="k", ls="--", lw=1.1, label=f"Ground truth = {x2_mean:.4f} A")
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("D3.2: KTC 95% cua mu_hat (Che do 2)")
    ax.set_ylabel("Dong dien (A)")
    ax.legend(fontsize=7.5); ax.grid(True, alpha=0.25)

    # Panel 3: Kiem tra WSS (delta_mu nua dau - nua sau)
    ax = axes[2]
    ax.errorbar([0, 1], [wss_m2["d_mu"], wss_m3["d_mu"]],
                yerr=[wss_m2["ci_diff"], wss_m3["ci_diff"]],
                fmt="o", color="C0", capsize=6, lw=1.6, ms=6)
    ax.axhline(0, color="k", ls="--", lw=1.0, label="delta_mu = 0 (WSS)")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Che do 2\n(WSS)", "Che do 3\n(Khong dung)"], fontsize=8.5)
    ax.set_title("D3.3: Hieu trung binh nua dau - nua sau")
    ax.set_ylabel("mu_1 - mu_2 (A)")
    ax.legend(fontsize=7.5); ax.grid(True, alpha=0.25)

    fig.tight_layout()
    out_path = "figures/fig_d3_stats_ci.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  -> Da luu hinh: {out_path}\n")
    return dict(m2_ci_freq=ci_freq, kappa_freq=kappa_freq, wss_m2=wss_m2)


# ==============================================================================
# D4 - UOC LUONG ACF (NHIEU THUAN, CHE DO 2, PHAN DU SAU KHU GON)
# ==============================================================================
def d4_acf_analysis(data, calib):
    """D4 - Uoc luong ACF chia N cho 3 tin hieu:
    (a) 10 s I=0 dau tien cua Che do 1;
    (b) Che do 2 tru trung binh;
    (c) Phan du Che do 2 sau khi khu gon tai f_c va 2*f_c.
    """
    print("=" * 76)
    print("D4 - UOC LUONG HAM TU TUONG QUAN (ACF)")
    print("=" * 76)
    VQ, SI = calib["V_Q"], calib["S_I"]
    to_amp = lambda y: (y - VQ) / SI

    # (a) 10 s I=0 dau tien cua Che do 1 (bo 50 ms dau)
    segs = _extract_mode1_segments(data[1], skip_s=0.050)
    z1_0 = to_amp(segs[0]["y"])
    fs = data[1]["fs"]

    # (b) Che do 2 tru trung binh
    z2 = to_amp(data[2]["y"])
    z2_zm = z2 - z2.mean()

    # (c) Khu gon co gop o Che do 2 bang cach tim dinh pho f_c tu chinh du lieu z2
    fw, Pw = U.welch_psd(z2, fs, 8192)
    mask_fc = (fw > 200.0) & (fw < 600.0)
    fc_hat = float(fw[mask_fc][np.argmax(Pw[mask_fc])])
    # Loc chan dai (notch) bac 2 tai fc_hat va 2*fc_hat voi Q=12, dung filtfilt
    b1, a1 = signal.iirnotch(fc_hat, Q=12.0, fs=fs)
    b2, a2 = signal.iirnotch(2.0 * fc_hat, Q=12.0, fs=fs)
    z2_res = signal.filtfilt(b1, a1, z2_zm)
    z2_res = signal.filtfilt(b2, a2, z2_res)

    items = [
        ("(a) 10s I=0 (nhieu thuan)", z1_0, 10.0),
        ("(b) Che do 2 (co gon)", z2_zm, 40.0),
        (f"(c) Phan du Che do 2 (khu gon {fc_hat:.1f} Hz)", z2_res, 150.0),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(11.5, 7.6))
    for row, (title, sig, short_ms) in enumerate(items):
        N = len(sig)
        maxlag = N // 4
        _, rho = U.acf_biased(sig, maxlag)
        lags_ms = np.arange(maxlag + 1) / fs * 1e3
        band = 1.96 / np.sqrt(N)
        n_short = int(short_ms * fs / 1e3) + 1

        # In thong so tom tat
        far_exceed = float(np.mean(np.abs(rho[maxlag // 2:]) > band) * 100)
        print(f"  {title:42s} | N={N:5d}, maxlag={maxlag:5d} ({lags_ms[-1]:.0f} ms) | "
              f"rho[1..3]={np.round(rho[1:4], 3)} | lag xa > band: {far_exceed:.1f}%")

        # Cot trai: phong to lag ngan
        ax_l = axes[row, 0]
        marker = ".-" if row == 0 else "-"
        ax_l.plot(lags_ms[:n_short], rho[:n_short], marker, color="C0", lw=0.9, ms=4)
        ax_l.axhline(0, color="k", lw=0.6)
        ax_l.axhline(band, color="C3", ls="--", lw=0.8, label=f"+-1.96/sqrt(N) = +-{band:.4f}")
        ax_l.axhline(-band, color="C3", ls="--", lw=0.8)
        ax_l.set_title(f"{title} - Lag ngan (0..{short_ms:.0f} ms)")
        ax_l.set_xlabel("lag (ms)"); ax_l.set_ylabel("rho_hat[l]")
        ax_l.legend(fontsize=7, loc="upper right"); ax_l.grid(True, alpha=0.25)

        # Cot phai: toan dai |l| <= N/4
        ax_r = axes[row, 1]
        ax_r.plot(lags_ms, rho, color="C0", lw=0.55)
        ax_r.axhline(0, color="k", lw=0.6)
        ax_r.axhline(band, color="C3", ls="--", lw=0.8)
        ax_r.axhline(-band, color="C3", ls="--", lw=0.8)
        ax_r.set_title(f"{title} - Toan dai l <= N/4 ({lags_ms[-1]/1e3:.2f} s)")
        ax_r.set_xlabel("lag (ms)"); ax_r.set_ylabel("rho_hat[l]")
        ax_r.set_ylim(-0.25, 1.05) if row != 2 else ax_r.set_ylim(-0.35, 1.05)
        ax_r.grid(True, alpha=0.25)

    fig.tight_layout()
    out_path = "figures/fig_d4_acf.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  -> Da luu hinh: {out_path}\n")
    return dict(fc_hat=fc_hat, out_path=out_path)


# ==============================================================================
# D5 - PSD BANG WELCH, CHON L VA DOI CHIEU LY THUYET M1
# ==============================================================================
def _welch_with_se(x, fs, L, overlap=0.5):
    """Tinh Welch PSD (Hann, chong 50%) va sai so chuan giua K doan: SE(f) = std_k(P_k)/sqrt(K)."""
    x0 = x - x.mean()
    step = int(L * (1.0 - overlap))
    win = np.hanning(L)
    # Chuan hoa nang luong cua so giong scipy.signal.welch (scaling='density')
    U_win = np.sum(win ** 2)
    starts = range(0, len(x0) - L + 1, step)
    P_segs = []
    for s in starts:
        seg = x0[s:s + L] * win
        X = np.fft.rfft(seg)
        P = (np.abs(X) ** 2) / (fs * U_win)
        P[1:-1] *= 2.0  # PSD mot phia
        P_segs.append(P)
    P_segs = np.array(P_segs)
    K = len(P_segs)
    f = np.fft.rfftfreq(L, 1.0 / fs)
    P_mean = P_segs.mean(axis=0)
    P_se = P_segs.std(axis=0, ddof=1) / np.sqrt(K)
    return f, P_mean, P_se, P_segs, K


def d5_welch_psd(data, calib):
    """D5 - Welch PSD voi L in {256, 2048, 8192}, thanh sai so cho L=2048,
    ly do chon L, va doi chieu cong suat theo dai voi M1 (m1_expected_psd.py).
    """
    print("=" * 76)
    print("D5 - PSD BANG WELCH, CHON L & DOI CHIEU LY THUYET M1")
    print("=" * 76)
    VQ, SI = calib["V_Q"], calib["S_I"]
    d2 = data[2]
    fs = d2["fs"]
    z2 = (d2["y"] - VQ) / SI

    # 1-2. Welch cho L in {256, 2048, 8192} va thanh sai so cho L=2048
    f256, P256 = U.welch_psd(z2, fs, 256)
    f2048, P2048, se2048, P_segs_2048, K2048 = _welch_with_se(z2, fs, 2048)
    f8192, P8192, se8192, P_segs_8192, K8192 = _welch_with_se(z2, fs, 8192)

    rel_se_2048 = float(np.median(se2048 / P2048) * 100)
    print(f"  Welch L=256  (df={fs/256:.2f} Hz) | L=2048 (K={K2048}, df={fs/2048:.2f} Hz, median SE~{rel_se_2048:.1f}%) | "
          f"L=8192 (K={K8192}, df={fs/8192:.2f} Hz)")

    # 4. Doi chieu voi PSD ly thuyet M1 (CHI dung tham so config cho cot ly thuyet de cham diem)
    f_th = np.linspace(0.05, fs / 2.0, 20000)
    w_ss = S_sim.steady_state(C.MODE2["tauL0"])
    mu_th = w_ss[0]
    fc_th = C.MOTOR["N_seg"] * w_ss[1] / (2.0 * np.pi)
    var_x_th = _trapz(M1.S_x(f_th), f_th)
    P_th = M1.S_x(f_th) + M1.ripple(f_th, mu_th, var_x_th, fc_th) + M1.S_c(f_th) + M1.S_v(f_th)

    bands = [(0.5, 5.0), (5.0, 50.0), (100.0, 300.0), (360.0, 390.0), (900.0, 1200.0)]
    print("\n  [Bang doi chieu cong suat theo dai (A^2)]")
    print("   Dai (Hz)     | Ly thuyet M1 | Welch L=8192 (+- SE)        | Sai lech (%)")
    band_rows = []
    for a, b in bands:
        mk = (f_th >= a) & (f_th < b)
        mw = (f8192 >= a) & (f8192 < b)
        pw_th = float(_trapz(P_th[mk], f_th[mk]))
        # Cong suat tung doan tren dai -> tinh trung binh va SE
        p_seg_band = _trapz(P_segs_8192[:, mw], f8192[mw], axis=1)
        pw_est = float(p_seg_band.mean())
        pw_se = float(p_seg_band.std(ddof=1) / np.sqrt(K8192))
        diff_pct = (pw_est - pw_th) / pw_th * 100.0
        band_rows.append((f"{a:g}-{b:g}", pw_th, pw_est, pw_se, diff_pct))
        print(f"   {a:5.1f}-{b:6.1f} | {pw_th:12.3e} | {pw_est:10.3e} +- {pw_se:9.2e} | {diff_pct:+7.1f}%")

    # Ve hinh D5
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1.35, 1.0]})

    # Panel 1: Chồng 3 đường Welch + dải sai số L=2048 + Lý thuyết M1
    ax = axes[0]
    ax.semilogy(f256, P256, "C0", lw=0.9, alpha=0.85, label="Welch L=256 (df=9.77 Hz)")
    ax.semilogy(f8192, P8192, "C2", lw=0.7, alpha=0.75, label="Welch L=8192 (df=0.31 Hz)")
    ax.semilogy(f2048, P2048, "C1", lw=1.0, label="Welch L=2048 (df=1.22 Hz)")
    low_b = np.maximum(P2048 - se2048, 1e-10)
    high_b = P2048 + se2048
    ax.fill_between(f2048, low_b, high_b, color="C1", alpha=0.25, label="L=2048 +- 1 SE")
    ax.semilogy(f_th, P_th, "k--", lw=1.0, label="Ly thuyet M1 (doi chieu)")
    ax.set_ylim(1e-9, 1e-1)
    ax.set_xlabel("f (Hz)  [w = 2*pi*f/2500]")
    ax.set_ylabel("PSD (A^2/Hz)")
    ax.set_title("D5.1: Welch PSD theo chieu dai doan L")
    ax.legend(fontsize=7.2, loc="upper right"); ax.grid(True, alpha=0.25)

    # Panel 2: Phong to quanh buou gon f_c de thay ro ly do chon L=8192
    ax = axes[1]
    m_z1 = (f256 >= 355) & (f256 <= 395)
    m_z2 = (f2048 >= 355) & (f2048 <= 395)
    m_z3 = (f8192 >= 355) & (f8192 <= 395)
    m_zt = (f_th >= 355) & (f_th <= 395)
    ax.semilogy(f_th[m_zt], P_th[m_zt], "k--", lw=1.2, label="Ly thuyet M1 (be rong ~1 Hz)")
    ax.semilogy(f8192[m_z3], P8192[m_z3], "C2.-", lw=1.0, ms=3, label="L=8192 (df=0.31 Hz)")
    ax.semilogy(f2048[m_z2], P2048[m_z2], "C1s-", lw=0.9, ms=3, label="L=2048 (df=1.22 Hz)")
    ax.semilogy(f256[m_z1], P256[m_z1], "C0^-", lw=0.9, ms=3.5, label="L=256 (df=9.77 Hz)")
    ax.set_xlabel("f (Hz)"); ax.set_ylabel("PSD (A^2/Hz)")
    ax.set_title(f"D5.2: Phong to buou gon quanh f_c ~ {fc_th:.1f} Hz")
    ax.legend(fontsize=7.2); ax.grid(True, alpha=0.25)

    fig.tight_layout()
    out_path = "figures/fig_d5_welch_psd.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  -> Da luu hinh: {out_path}\n")
    return dict(band_rows=band_rows, out_path=out_path)


# ==============================================================================
# D7 - UOC LUONG THAM SO MO HINH CAM BIEN & KIEM CHUNG MO HINH M1 HIEU CHUAN
# ==============================================================================
def d7_calibrated_model_check(data, calib):
    """D7 - Uoc luong cac tham so dong hoc cua mo hinh cam bien (nhieu nen AR(2),
    troi nhiet bac 1, dong tai AR(1), buou gon Gauss) kem sai so; dung PSD mo hinh
    M1 da hieu chuan doi chieu voi du lieu va kiem dinh cac gia thiet cua M1.
    """
    print("=" * 76)
    print("D7 - THAM SO MO HINH CAM BIEN & KIEM CHUNG MO HINH M1 HIEU CHUAN")
    print("=" * 76)
    VQ, SI = calib["V_Q"], calib["S_I"]
    to_amp = lambda y: (y - VQ) / SI

    # 1. Tham so nhieu nen tai I=0 (10 s dau Che do 1, bo 50 ms) -> sigma_v^2 va AR(2)
    segs = _extract_mode1_segments(data[1], skip_s=0.050)
    fs = data[1]["fs"]
    v_V = segs[0]["y"] - segs[0]["y"].mean()
    v_A = v_V / SI
    N_v = len(v_A)
    mc_v = U.mean_ci(v_A)
    Neff_v = mc_v["Neff"]
    var_v_V = float(np.var(v_V))
    var_v_A = float(np.var(v_A))
    se_var_v_A = float(var_v_A * np.sqrt(2.0 / Neff_v))

    # LS cho AR(2): v[n] = a1*v[n-1] + a2*v[n-2] + e[n]
    H_ar2 = np.column_stack([v_A[1:-1], v_A[:-2]])
    y_ar2 = v_A[2:]
    th_ar2, cov_ar2, r_ar2 = U.ls_fit(H_ar2, y_ar2)
    se_ar2 = np.sqrt(np.diag(cov_ar2))
    sigma_e2_A = float(np.var(r_ar2, ddof=2))
    print(f"  [Nhieu nen I=0]  sigma_v = {np.sqrt(var_v_V)*1e3:.3f} mV ({np.sqrt(var_v_A)*1e3:.3f} mA) | "
          f"var_v = ({var_v_A*1e6:.2f} +- {1.96*se_var_v_A*1e6:.2f}) uA^2")
    print(f"                   AR(2) LS: a1 = {th_ar2[0]:+.4f} +- {1.96*se_ar2[0]:.4f}, "
          f"a2 = {th_ar2[1]:+.4f} +- {1.96*se_ar2[1]:.4f}, sigma_e = {np.sqrt(sigma_e2_A)*1e3:.3f} mA")

    # 2. Uoc luong tham so troi nhiet tu 9 diem I=0 (Che do 1)
    zero_segs = [s for s in segs if s["I_ref"] == 0.0]
    t0_pts = np.array([s["t_mid"] for s in zero_segs])
    dV0_pts = np.array([s["stats"]["mean"] - zero_segs[0]["stats"]["mean"] for s in zero_segs])
    se0_pts = np.array([s["stats"]["ci"] / 1.96 for s in zero_segs])

    # Dung day I_ref(t) cua Che do 1 o tan so fs (biet chinh xac tu thiet ke D1)
    I_seq_fs = np.concatenate([np.full(int(round(T * fs)), I) for I, T in C.MODE1["seq"]])
    t_fs = np.arange(len(I_seq_fs)) / fs
    dt_s = 1.0 / fs
    drive_i2 = I_seq_fs ** 2

    # (2a) Fit tuyen tinh toc do phat nhiet g_rate = kP*TC / tau_th (khi tau_th >> T)
    int_i2 = np.cumsum(drive_i2) * dt_s
    int_i2_pts = np.interp(t0_pts, t_fs, int_i2)
    H_rate = int_i2_pts[:, None]
    th_rate, cov_rate, _ = U.ls_fit(H_rate, dV0_pts)
    se_rate = float(np.sqrt(cov_rate[0, 0]))

    # (2b) Fit phi tuyen 2 tham so (g_th = kP*TC, tau_th) voi WLS
    def _thermal_model(params):
        g_th, tau_th = params
        a = np.exp(-dt_s / tau_th)
        dV_sim = signal.lfilter([g_th * (1.0 - a)], [1.0, -a], drive_i2)
        return (np.interp(t0_pts, t_fs, dV_sim) - dV0_pts) / se0_pts

    res_opt = optimize.least_squares(_thermal_model, x0=[0.35e-3, 25.0], bounds=([1e-5, 2.0], [1e-2, 200.0]))
    g_th_hat, tau_th_hat = res_opt.x
    J_opt = res_opt.jac
    # Hiep phuong sai phi tuyen (chi dung khi duoc hoi tu tot)
    dof_th = max(len(t0_pts) - 2, 1)
    s2_th = float(np.sum(res_opt.fun ** 2) / dof_th)
    cov_th = s2_th * np.linalg.pinv(J_opt.T @ J_opt)
    se_g_th, se_tau_th = np.sqrt(np.diag(cov_th))
    g_th_true = C.SENSOR["kP"] * C.SENSOR["TC"]
    tau_th_true = C.SENSOR["tau_th"]
    print(f"  [Troi nhiet D1]  g_rate = kP*TC/tau_th = ({th_rate[0]*1e6:.3f} +- {1.96*se_rate*1e6:.3f}) uV/(A^2.s) "
          f"(that: {g_th_true/tau_th_true*1e6:.3f})")
    print(f"                   Fit 2 tham so: kP*TC = ({g_th_hat*1e3:.3f} +- {1.96*se_g_th*1e3:.3f}) mV/A^2 (that {g_th_true*1e3:.3f}), "
          f"tau_th = {tau_th_hat:.1f} +- {1.96*se_tau_th:.1f} s (that {tau_th_true:.1f})")

    # 3. Uoc luong tham so dong hoc tai & gon tu Che do 2
    z2 = to_amp(data[2]["y"])
    z2_zm = z2 - z2.mean()
    f8192, P8192, _, P_segs_8192, K8192 = _welch_with_se(z2, fs, 8192)

    # Tim f_c va 2*f_c, tinh cong suat & be rong buou Gauss
    def _fit_ripple_band(f_lo, f_hi):
        m = (f8192 >= f_lo) & (f8192 <= f_hi)
        f_b, P_b = f8192[m], P8192[m]
        # Tru nen uoc luong o hai mep dai
        bg = float(np.median(np.concatenate([P_b[:5], P_b[-5:]])))
        P_net = np.maximum(P_b - bg, 0.0)
        P_tot = float(_trapz(P_net, f_b))
        f_cent = float(_trapz(f_b * P_net, f_b) / P_tot)
        sd_f = float(np.sqrt(_trapz(((f_b - f_cent) ** 2) * P_net, f_b) / P_tot))
        # Sai so chuan cua cong suat va tam tan so tu K doan
        p_seg = _trapz(np.maximum(P_segs_8192[:, m] - bg, 0.0), f_b, axis=1)
        se_P = float(p_seg.std(ddof=1) / np.sqrt(K8192))
        fc_seg = np.array([_trapz(f_b * np.maximum(ps - bg, 0.0), f_b) / max(_trapz(np.maximum(ps - bg, 0.0), f_b), 1e-12)
                           for ps in P_segs_8192[:, m]])
        se_fc = float(fc_seg.std(ddof=1) / np.sqrt(K8192))
        return dict(fc=f_cent, se_fc=se_fc, sd=sd_f, P=P_tot, se_P=se_P)

    m_fc = (f8192 > 200.0) & (f8192 < 600.0)
    fc_peak = float(f8192[m_fc][np.argmax(P8192[m_fc])])
    rip1 = _fit_ripple_band(fc_peak - 15.0, fc_peak + 15.0)
    rip2 = _fit_ripple_band(2 * fc_peak - 20.0, 2 * fc_peak + 20.0)
    print(f"  [Gon co gop M2]  Hai 1: f_c = {rip1['fc']:.3f} +- {1.96*rip1['se_fc']:.3f} Hz, "
          f"sd = {rip1['sd']:.2f} Hz, P1 = ({rip1['P']*1e3:.3f} +- {1.96*rip1['se_P']*1e3:.3f}) mA^2")
    print(f"                   Hai 2: 2f_c = {rip2['fc']:.3f} +- {1.96*rip2['se_fc']:.3f} Hz, "
          f"sd = {rip2['sd']:.2f} Hz, P2 = ({rip2['P']*1e3:.3f} +- {1.96*rip2['se_P']*1e3:.3f}) mA^2")

    # Khu gon -> fit AR(1) cho dong tai: z_res[n] = phi*z_res[n-1] + w[n]
    b1, a1 = signal.iirnotch(rip1["fc"], Q=12.0, fs=fs)
    b2, a2 = signal.iirnotch(rip2["fc"], Q=12.0, fs=fs)
    z2_res = signal.filtfilt(b2, a2, signal.filtfilt(b1, a1, z2_zm))
    H_ar1 = z2_res[:-1, None]
    th_ar1, cov_ar1, r_ar1 = U.ls_fit(H_ar1, z2_res[1:])
    phi_hat = float(th_ar1[0])
    # Hieu chinh sai so chuan cho AR(1)
    se_phi = float(np.sqrt(cov_ar1[0, 0]))
    # Tinh lai phuong sai dong tai tu cong suat dai thap (< 30 Hz) sau khi tru nen
    m_low = (f8192 > 0) & (f8192 <= 30.0)
    var_x_est = float(np.maximum(_trapz(P8192[m_low], f8192[m_low]) - 30.0 * 1.1e-7, 1e-7))
    tau_m_est = -dt_s / np.log(min(phi_hat, 0.99999))
    print(f"  [Dong tai AR(1)] phi_hat = {phi_hat:.4f} +- {1.96*se_phi:.4f} | "
          f"var_x_est (<30 Hz) = {var_x_est*1e4:.3f}e-4 A^2 (sigma_x ~ {np.sqrt(var_x_est)*1e3:.2f} mA)")

    # 4. Dung PSD cua mo hinh M1 da hieu chuan (HOAN TOAN tu tham so uoc luong)
    f_grid = f8192[1:]  # bo DC
    z_ejw = np.exp(-1j * 2.0 * np.pi * f_grid / fs)
    S_noise_calib = (2.0 * sigma_e2_A / fs) / (np.abs(1.0 - th_ar2[0] * z_ejw - th_ar2[1] * (z_ejw ** 2)) ** 2)

    # Thanh phan tai OU bac 1 tu dai tan thap (f_c_load uoc luong tu nua cong suat dai thap)
    P_low_0 = float(np.mean(P8192[(f8192 > 0) & (f8192 <= 2.5)]))
    # S_load(f) = P_low_0 / (1 + (f/f_load)^2), tich phan = P_low_0 * f_load * pi/2 = var_x_est
    f_load_est = var_x_est / (P_low_0 * np.pi / 2.0)
    S_load_calib = P_low_0 / (1.0 + (f_grid / f_load_est) ** 2)

    def _gauss(f, fc_v, sd_v, P_v):
        return P_v * np.exp(-0.5 * ((f - fc_v) / sd_v) ** 2) / (np.sqrt(2.0 * np.pi) * sd_v)

    S_rip_calib = _gauss(f_grid, rip1["fc"], rip1["sd"], rip1["P"]) + _gauss(f_grid, rip2["fc"], rip2["sd"], rip2["P"])
    P_m1_calib = S_load_calib + S_rip_calib + S_noise_calib

    # 5. Ve hinh D7
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1.3, 1.0]})

    ax = axes[0]
    ax.semilogy(f_grid, P8192[1:], color="0.65", lw=0.7, label="Welch L=8192 (du lieu do M2)")
    ax.semilogy(f_grid, P_m1_calib, "C0", lw=1.2, label="M1 hieu chuan (tu tham so uoc luong)")
    ax.semilogy(f_grid, S_noise_calib, "C2--", lw=0.9, label="Nhieu nen AR(2) uoc luong")
    ax.semilogy(f_grid, S_load_calib, "C1--", lw=0.9, label=f"Tai tan thap (f_load={f_load_est:.1f} Hz)")
    ax.set_ylim(1e-9, 1e-1)
    ax.set_xlabel("f (Hz)"); ax.set_ylabel("PSD (A^2/Hz)")
    ax.set_title("D7.1: Mo hinh M1 hieu chuan vs Welch PSD thuc nghiem")
    ax.legend(fontsize=7.2); ax.grid(True, alpha=0.25)

    # Panel 2: Fit mo hinh troi nhiet bac 1 tren 9 diem I=0 cua Che do 1
    ax = axes[1]
    a_hat = np.exp(-dt_s / tau_th_hat)
    dV_fit = signal.lfilter([g_th_hat * (1.0 - a_hat)], [1.0, -a_hat], drive_i2)
    ax.errorbar(t0_pts, dV0_pts * 1e3, yerr=1.96 * se0_pts * 1e3,
                fmt="o", color="C0", capsize=3, ms=4.5, label="Doan I=0 (+-KTC 95%)")
    ax.plot(t_fs, dV_fit * 1e3, "C1-", lw=1.2,
            label=f"Fit bac 1: kP*TC={g_th_hat*1e3:.2f} mV/A^2, tau={tau_th_hat:.1f}s")
    ax.plot(t_fs, th_rate[0] * int_i2 * 1e3, "C2--", lw=1.0,
            label=f"Tuyen tinh: {th_rate[0]*1e6:.2f} uV/(A^2.s)")
    ax.set_xlabel("t (s)"); ax.set_ylabel("Troi nhiet (mV)")
    ax.set_title("D7.2: Uoc luong tham so troi nhiet (Che do 1)")
    ax.legend(fontsize=7.2); ax.grid(True, alpha=0.25)

    fig.tight_layout()
    out_path = "figures/fig_d7_calibrated_model.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  -> Da luu hinh: {out_path}\n")
    return dict(
        sigma_v_A=float(np.sqrt(var_v_A)), se_var_v_A=se_var_v_A,
        ar2=th_ar2, se_ar2=se_ar2, sigma_e_A=float(np.sqrt(sigma_e2_A)),
        g_rate=float(th_rate[0]), se_rate=se_rate,
        g_th=float(g_th_hat), se_g_th=float(se_g_th),
        tau_th=float(tau_th_hat), se_tau_th=float(se_tau_th),
        rip1=rip1, rip2=rip2, phi_load=phi_hat, se_phi=se_phi,
        var_x_est=var_x_est, f_load_est=f_load_est, out_path=out_path
    )


if __name__ == "__main__":
    data = load_data()
    d2_raw_check(data)
    calib = d6_calibration(data)
    d3_basic_stats(data, calib)
    d4_acf_analysis(data, calib)
    d5_welch_psd(data, calib)
    d7_calibrated_model_check(data, calib)
