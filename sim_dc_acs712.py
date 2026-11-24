"""sim_dc_acs712.py - Mo phong dong co DC choi than + ACS712-05B + loc analog + ADC 12 bit.

Khoi A (dong co): La di/dt + Ra i + Ke w = Us ; J dw/dt + B w = Kt i - (tauL0 + d_tauL)
  -> roi rac hoa CHINH XAC (ZOH, ma tran mu) o F_SIM, chay bang lfilter (nhanh);
     ham rk4_reference() dung de doi chieu tren mot doan ngan.
  Gon chuyen mach nhan: i_a = i_avg [1 + g1 cos(th_c+p1) + g2 cos(2 th_c+p2)], dth_c/dt = N_seg w
Khoi B (cam bien): u = 0.5Vcc + dV0 + dV_th(t) + S*i_a + n_hall + n_emi
  -> loc RC chan FILTER (RF, CF) -> Sallen-Key bac 2 -> lay mau F_S -> luong tu Q12.

Ground truth: x[n] = i_avg(n Ts) (dong trung binh, KHONG gom gon, KHONG loc).
"""
import numpy as np
from scipy import signal, linalg
import config as C


# ------------------------------------------------------------------ Khoi A
def motor_ss():
    m = C.MOTOR
    A = np.array([[-m["Ra"] / m["La"], -m["Ke"] / m["La"]],
                  [m["Kt"] / m["J"], -m["B"] / m["J"]]])
    B = np.array([[1 / m["La"], 0.0],
                  [0.0, -1 / m["J"]]])          # u = [Us, tauL]
    return A, B


def steady_state(tauL):
    A, B = motor_ss()
    return -np.linalg.solve(A, B @ np.array([C.MOTOR["Us"], tauL]))  # [i, w]


def ou_process(n, dt, sigma, tau, rng):
    """Ornstein-Uhlenbeck roi rac chinh xac: AR(1) voi a = exp(-dt/tau)."""
    a = np.exp(-dt / tau)
    e = rng.standard_normal(n) * sigma * np.sqrt(1 - a * a)
    x = signal.lfilter([1.0], [1.0, -a], e)
    return x


def simulate_motor(tauL0, rng, T=None):
    """tauL0: mang mo-men tai danh dinh theo luoi F_SIM. Tra ve dict cac tin hieu o F_SIM."""
    dt = 1.0 / C.F_SIM
    n = len(tauL0)
    dtau = ou_process(n, dt, C.LOAD_NOISE["sigma"], C.LOAD_NOISE["tau"], rng)
    tauL = tauL0 + dtau
    A, B = motor_ss()
    # ZOH chinh xac
    M = linalg.expm(np.block([[A, B], [np.zeros((2, 4))]]) * dt)
    Ad, Bd = M[:2, :2], M[:2, 2:]
    x0 = steady_state(tauL0[0])
    U = np.column_stack([np.full(n, C.MOTOR["Us"]), tauL])
    # x[k+1] = Ad x[k] + Bd u[k]  -> giai bang lfilter tren tung kenh (he tuyen tinh)
    y = _fast_lsim(Ad, Bd, U, x0)
    i_avg, w = y[:, 0], y[:, 1]
    m = C.MOTOR
    th_c = m["N_seg"] * np.cumsum(w) * dt
    p1, p2 = rng.uniform(0, 2 * np.pi, 2)          # pha ngau nhien ban dau
    i_a = i_avg * (1 + m["gamma1"] * np.cos(th_c + p1) + m["gamma2"] * np.cos(2 * th_c + p2))
    return dict(i_avg=i_avg, w=w, i_a=i_a, tauL=tauL, dtau=dtau, th_c=th_c)


def _fast_lsim(Ad, Bd, U, x0):
    """x[k+1]=Ad x[k]+Bd u[k], x[0]=x0 = trang thai xac lap voi u[0].
    Viet x = x0 + dx, du = u - u[0]: dx[0]=0 -> moi kenh vao-ra la mot ham truyen,
    giai bang lfilter (vong lap C, nhanh). Chinh xac nhu dlsim."""
    dU = U - U[0]
    y = np.tile(x0, (U.shape[0], 1)).astype(float)
    for out in range(2):
        for inp in range(2):
            num, den = signal.ss2tf(Ad, Bd[:, [inp]], np.eye(2)[[out]], np.zeros((1, 1)))
            y[:, out] += signal.lfilter(num[0], den, dU[:, inp])
    return y


def rk4_reference(tauL, x0, dt):
    """RK4 kinh dien de doi chieu (chay cham, chi dung cho doan ngan)."""
    A, B = motor_ss()
    x = np.array(x0, float)
    out = np.empty((len(tauL), 2))
    for k, tl in enumerate(tauL):
        out[k] = x
        u = np.array([C.MOTOR["Us"], tl])
        f = lambda s: A @ s + B @ u
        k1 = f(x); k2 = f(x + dt / 2 * k1); k3 = f(x + dt / 2 * k2); k4 = f(x + dt * k3)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return out


# ------------------------------------------------------------------ Khoi B
def thermal_offset(i_a, dt, T0=None):
    s = C.SENSOR
    a = np.exp(-dt / s["tau_th"])
    drive = s["kP"] * i_a ** 2                    # dT_ss
    zi = None if T0 is None else [a * T0]
    dT = signal.lfilter([1 - a], [1, -a], drive, zi=zi)
    dT = dT[0] if zi is not None else dT
    return s["TC"] * dT, dT


def analog_filters():
    fs = C.F_SIM
    f = C.FILTER
    wc1 = 1.0 / (f["RF"] * f["CF"])
    b1, a1 = signal.bilinear([wc1], [1, wc1], fs)
    w2 = 2 * np.pi * f["f_sk"]
    b2, a2 = signal.bilinear([w2 ** 2], [1, w2 / f["Q_sk"], w2 ** 2], fs)
    return (b1, a1), (b2, a2)


def sensor_chain(i_a, rng, t, dV_th):
    s, e, adc = C.SENSOR, C.EMI, C.ADC
    n = len(i_a)
    # nhieu Hall: trang o F_SIM, chuan hoa sao cho trong bang tuong duong datasheet (CF=47nF, bac 1)
    fc_ds = 1 / (2 * np.pi * C.FILTER["RF"] * 47e-9)
    N0 = s["sigma_hall_2k"] ** 2 / (np.pi / 2 * fc_ds)          # V^2/Hz (mot phia)
    n_hall = rng.standard_normal(n) * np.sqrt(N0 * C.F_SIM / 2)
    n_emi = e["A"] * np.cos(2 * np.pi * e["f_smps"] * t + rng.uniform(0, 2 * np.pi))
    u_raw = 0.5 * s["Vcc"] + s["dV0"] + dV_th + s["S_true"] * i_a + n_hall + n_emi
    (b1, a1), (b2, a2) = analog_filters()
    u = signal.lfilter(b1, a1, u_raw - u_raw[0], ) + u_raw[0]
    if C.FILTER["use_sallen_key"]:
        u = signal.lfilter(b2, a2, u - u[0]) + u[0]
    u_s = u[::C.DECIM]                                           # lay mau F_S
    lsb = adc["Vref"] / 2 ** adc["bits"]
    u_s = u_s + rng.standard_normal(len(u_s)) * adc["sigma_adc_lsb"] * lsb
    code = np.clip(np.round(u_s / lsb), 0, 2 ** adc["bits"] - 1).astype(int)
    return dict(u_raw=u_raw, u_an=u, code=code, y=code * lsb, lsb=lsb)


# ------------------------------------------------------------------ Kich ban
def _grid(T):
    n = int(round(T * C.F_SIM))
    return np.arange(n) / C.F_SIM


def tau_profile(t, profile):
    tau = np.zeros_like(t)
    for k, (t0, v) in enumerate(profile):
        tau[t >= t0] = v
    # doan doc: tu moc thu 4 den moc thu 5 (xem config)
    (ta, va), (tb, vb) = profile[3], profile[4]
    m = (t >= ta) & (t < tb)
    tau[m] = va + (vb - va) * (t[m] - ta) / (tb - ta)
    return tau


def run_mode(mode, seed=C.SEED):
    rng = np.random.default_rng(seed + 1000 * mode)
    dt = 1 / C.F_SIM
    if mode == 1:
        i_a = np.concatenate([np.full(int(T * C.F_SIM), I) for I, T in C.MODE1["seq"]])
        i_avg = i_a.copy()
        t = np.arange(len(i_a)) * dt
        dV_th, dT = thermal_offset(i_a, dt, T0=0.0)
        mot = dict(i_avg=i_avg, i_a=i_a)
    else:
        if mode == 2:
            t = _grid(C.MODE2["T"]); tau0 = np.full_like(t, C.MODE2["tauL0"])
        else:
            t = _grid(C.MODE3["T"]); tau0 = tau_profile(t, C.MODE3["profile"])
        mot = simulate_motor(tau0, rng)
        mot["tauL0"] = tau0
        i_ss = steady_state(tau0[0])[0]
        g_r = 1 + (C.MOTOR["gamma1"] ** 2 + C.MOTOR["gamma2"] ** 2) / 2   # E[i_a^2] = i^2 (1 + g1^2/2 + g2^2/2)
        T0 = C.SENSOR["kP"] * i_ss ** 2 * g_r if mode == 2 else 0.0   # che do 2: da nong san (WSS)
        dV_th, dT = thermal_offset(mot["i_a"], dt, T0=T0)
    sen = sensor_chain(mot["i_a"], rng, t, dV_th)
    k = slice(None, None, C.DECIM)
    return dict(mode=mode, t=t[k], x=mot["i_avg"][k], i_a=mot["i_a"][k],
                w=mot.get("w", np.zeros_like(t))[k], dV_th=dV_th[k], dT=dT[k],
                y=sen["y"], code=sen["code"], lsb=sen["lsb"],
                fs=C.F_S, full=dict(t=t, **mot, u_an=sen["u_an"]))


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    for m in (1, 2, 3):
        d = run_mode(m)
        np.savez_compressed(f"data/mode{m}.npz", t=d["t"], y=d["y"], code=d["code"], x=d["x"],
                            i_a=d["i_a"], w=d["w"], dV_th=d["dV_th"], fs=d["fs"])
        print(f"mode {m}: N={len(d['y'])}, x mean={d['x'].mean():.3f} A, y mean={d['y'].mean():.4f} V")
