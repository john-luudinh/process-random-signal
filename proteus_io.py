"""proteus_io.py - Trao doi du lieu Python <-> Proteus (va ngspice de kiem chung truoc).

Lenh:
  python proteus_io.py export 2 0.5          # file cho generator FILE (+ ban 0.2 s, file thu, WAV du phong)
  python proteus_io.py netlist 2 0.5         # netlist ngspice cung mach (de doi chieu / tham khao noi day)
  python proteus_io.py compare <file> 2 0.5 [VOUT|FILTER] [ac]   # so du lieu Proteus voi Python ('ac': bo DC, khi dung WAV)
  python proteus_io.py compare_ac <file>     # so dap ung tan so Proteus (f, dB) voi ly thuyet

Mach (khop config.py: RC 936 Hz + Sallen-Key 600 Hz, Q=0.707):
  IN --RF 1.7k-- FILTER --(CF 100n)-- GND ; FILTER -> dem U1 (follower) -> BUF
  BUF --R1 18.7k-- A --R2 18.7k-- B ; C1 20n (2x10n song song) tu A ve VOUT ; C2 10n tu B ve GND
  U2: (+) = B, (-) = VOUT, out = VOUT  (Sallen-Key unity gain)
"""
import sys
import wave
import numpy as np
import config as C
import sim_dc_acs712 as S

OUT = "proteus"


def _raw_from_run(mode):
    """Chay lai che do voi CUNG seed nhu run_mode -> cung realization; tra ve u_raw, u_an (100 kHz)."""
    rng = np.random.default_rng(C.SEED + 1000 * mode)
    dt = 1 / C.F_SIM
    if mode == 1:
        raise ValueError("che do 1: dung nguon DC trong Proteus (I_ref * S_I + 2.518 V)")
    T = C.MODE2["T"] if mode == 2 else C.MODE3["T"]
    t = S._grid(T)
    tau0 = np.full_like(t, C.MODE2["tauL0"]) if mode == 2 else S.tau_profile(t, C.MODE3["profile"])
    mot = S.simulate_motor(tau0, rng)
    i_ss = S.steady_state(tau0[0])[0]
    g_r = 1 + (C.MOTOR["gamma1"] ** 2 + C.MOTOR["gamma2"] ** 2) / 2
    T0 = C.SENSOR["kP"] * i_ss ** 2 * g_r if mode == 2 else 0.0
    dV_th, _ = S.thermal_offset(mot["i_a"], dt, T0=T0)
    sen = S.sensor_chain(mot["i_a"], rng, t, dV_th)
    # u_an KHONG qua Sallen-Key (de so voi nut FILTER/BUF)
    (b1, a1), _ = S.analog_filters()
    from scipy import signal
    u_rc = signal.lfilter(b1, a1, sen["u_raw"] - sen["u_raw"][0]) + sen["u_raw"][0]
    return t, sen["u_raw"], sen["u_an"], u_rc


def export(mode=2, T=0.5):
    t, u_raw, u_an, u_rc = _raw_from_run(mode)
    n = int(T * C.F_SIM)
    tag = f"mode{mode}_{int(T*1000)}ms"
    np.savetxt(f"{OUT}/{tag}_uraw.txt", np.column_stack([t[:n], u_raw[:n]]), fmt="%.6e")
    np.savez_compressed(f"{OUT}/{tag}_python.npz", t=t[:n], u_raw=u_raw[:n], u_an=u_an[:n], u_rc=u_rc[:n])
    # file thu: buoc 2.5 V -> 3.0 V tai 1 ms, dai 5 ms (kiem tra Proteus doc dung dinh dang)
    tt = np.arange(0, 5e-3 + 1e-9, 1e-5)
    np.savetxt(f"{OUT}/test_step.txt", np.column_stack([tt, np.where(tt < 1e-3, 2.5, 3.0)]), fmt="%.6e")
    # WAV du phong cho generator AUDIO: chi phan xoay chieu, chuan hoa +-1; ghi he so bien do + DC
    ac = u_raw[:n] - u_raw[:n].mean()
    A = np.abs(ac).max()
    with wave.open(f"{OUT}/{tag}_ac.wav", "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(int(C.F_SIM))
        w.writeframes((ac / A * 32767).astype("<i2").tobytes())
    with open(f"{OUT}/{tag}_wav_info.txt", "w") as f:
        f.write(f"AUDIO generator: amplitude (dinh) = {A:.6f} V ; DC noi tiep = {u_raw[:n].mean():.6f} V\n")
    print(f"Da ghi {OUT}/{tag}_uraw.txt ({n} diem), {tag}_python.npz, test_step.txt, {tag}_ac.wav (A={A:.4f} V, DC={u_raw[:n].mean():.4f} V)")


def netlist(mode=2, T=0.5):
    tag = f"mode{mode}_{int(T*1000)}ms"
    f = C.FILTER
    txt = f"""* EE6423 - chuoi do ACS712 (R_F, C_F, dem) + Sallen-Key - cung mach voi Proteus
.model fsrc filesource (file="{tag}_uraw.txt" amploffset=[0] amplscale=[1] timeoffset=0 timescale=1 timerelative=false amplstep=false)
a1 %v([in]) fsrc
RF in filter {f['RF']:.0f}
CF filter 0 {f['CF']:.3e}
* U1: dem (op-amp ly tuong, follower)
EU1 buf 0 filter buf 1e6
R1 buf a 18.7k
R2 a b 18.7k
C1 a vout 20n
C2 b 0 10n
* U2: Sallen-Key unity gain
EU2 vout 0 b vout 1e6
.tran 10u {T} 0 10u
.control
run
wrdata spice_{tag}_out.txt v(vout) v(filter)
.endc
.end
"""
    open(f"{OUT}/chain_{tag}.cir", "w").write(txt)
    ac = """* EE6423 - dap ung tan so chuoi do
V1 in 0 dc 2.5 ac 1
RF in filter 1700
CF filter 0 100n
EU1 buf 0 filter buf 1e6
R1 buf a 18.7k
R2 a b 18.7k
C1 a vout 20n
C2 b 0 10n
EU2 vout 0 b vout 1e6
.ac dec 100 10 100k
.control
run
wrdata spice_ac_out.txt vdb(vout) vp(vout)
.endc
.end
"""
    open(f"{OUT}/chain_ac.cir", "w").write(ac)
    print(f"Da ghi {OUT}/chain_{tag}.cir va {OUT}/chain_ac.cir")


def read_graph(path):
    """Doc file text 2+ cot so (bo dong tieu de). Tra ve mang (N, k)."""
    rows = []
    with open(path, errors="ignore") as fh:
        for line in fh:
            parts = line.replace(",", " ").replace(";", " ").split()
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                continue
            if len(vals) >= 2:
                rows.append(vals)
    w = min(len(r) for r in rows)
    return np.array([r[:w] for r in rows])


def compare(path, mode=2, T=0.5, node="VOUT", col=1, ac=False):
    a = read_graph(path)
    tp, vp = a[:, 0], a[:, col]
    ref = np.load(f"{OUT}/mode{mode}_{int(T*1000)}ms_python.npz")
    yref = ref["u_an"] if node.upper() == "VOUT" else ref["u_rc"]
    ts = np.arange(0.01, min(tp[-1], ref["t"][-1]), 1 / C.F_S)   # bo 10 ms dau (khoi dong)
    vp_s, vr_s = np.interp(ts, tp, vp), np.interp(ts, ref["t"], yref)
    if ac:   # dung khi chay bang generator AUDIO (WAV chi co phan xoay chieu): bo thanh phan DC
        vp_s, vr_s = vp_s - vp_s.mean(), vr_s - vr_s.mean()
    e = vp_s - vr_s
    lsb = C.ADC["Vref"] / 2 ** C.ADC["bits"]
    rms = np.sqrt(np.mean(e ** 2))
    print(f"[{node}] sai lech tai cac thoi diem lay mau 2.5 kHz: RMS = {rms*1e3:.4f} mV = {rms/lsb:.3f} LSB; "
          f"max = {np.abs(e).max()*1e3:.4f} mV; lech TB = {e.mean()*1e3:.4f} mV -> {'DAT' if rms < lsb else 'CHUA DAT'} (< 1 LSB)")
    return rms


def H_theory(f):
    s = 2j * np.pi * f
    w = 1 / (18.7e3 * np.sqrt(20e-9 * 10e-9))
    return 1 / (1 + s * C.FILTER["RF"] * C.FILTER["CF"]) * w * w / (s * s + s * w / 0.70711 + w * w)


def compare_ac(path, fcol=0, dbcol=1):
    a = read_graph(path)
    f, db = a[:, fcol], a[:, dbcol]
    for fx in [50, 374.5, 749, 1100, 31400]:
        print(f"f = {fx:7.1f} Hz: do = {np.interp(fx, f, db):7.2f} dB, ly thuyet = {20*np.log10(abs(H_theory(fx))):7.2f} dB")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "export":
        export(int(sys.argv[2]), float(sys.argv[3]))
    elif cmd == "netlist":
        netlist(int(sys.argv[2]), float(sys.argv[3]))
    elif cmd == "compare":
        node = sys.argv[5] if len(sys.argv) > 5 else "VOUT"
        ac = len(sys.argv) > 6 and sys.argv[6].lower() == "ac"
        compare(sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), node, ac=ac)
    elif cmd == "compare_ac":
        compare_ac(sys.argv[2])
