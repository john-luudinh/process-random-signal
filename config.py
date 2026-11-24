"""config.py - Toan bo tham so mo phong EE6423 (dong co DC + ACS712-05B).

Moi tham so ghi kem nguon / ly do. Doi tham so o day, KHONG sua trong code khac.
"""
import numpy as np

SEED = 20252646            # co dinh random seed (ma HV) -> tai lap duoc

# ---------------- Thoi gian ----------------
F_SIM = 100_000.0          # Hz, buoc mo phong "lien tuc" (>> 31.4 kHz SMPS, >> 80 kHz? xem ghi chu)
F_S = 2_500.0              # Hz, tan so lay mau ADC
DECIM = int(F_SIM // F_S)  # 40

# ---------------- Khoi A: dong co DC choi than ----------------
# Gia tri dien hinh cho dong co 12 V co nho (~20-30 W); doi chieu datasheet dong co nhom chon.
MOTOR = dict(
    Us=12.0,        # V, dien ap nguon
    Ra=1.2,         # Ohm, dien tro phan ung
    La=2.5e-3,      # H, dien cam phan ung -> tau_e = La/Ra = 2.08 ms
    Ke=0.045,       # V.s/rad (= Kt, don vi SI)
    Kt=0.045,       # N.m/A
    J=4e-4,         # kg.m^2 (rotor + tai) -> tau_m = J*Ra/(Ke*Kt) = 0.237 s
    B=2e-4,         # N.m.s/rad, ma sat nhot
    N_seg=12,       # so phien co gop -> f_c = N_seg * f_rot
    gamma1=0.12,    # bien do tuong doi hai bac 1 cua gon chuyen mach
    gamma2=0.04,    # bien do tuong doi hai bac 2
)
# Bien dong tai ngau nhien delta_tau_L: qua trinh Ornstein-Uhlenbeck (mau, bang hep quanh 0)
LOAD_NOISE = dict(sigma=0.004, tau=0.02)   # N.m, s  (~5% tai danh dinh 0.08 N.m)

# ---------------- Khoi B: ACS712ELCTR-05B + chuoi do ----------------
SENSOR = dict(
    Vcc=5.0,               # V; dau ra ratiometric, diem 0 A = 0.5*Vcc
    S_nom=0.185,           # V/A, datasheet (min 0.180, max 0.190)
    S_true=0.1825,         # V/A, gia tri "that" cua chip mo phong (trong dai datasheet)
    dV0=0.018,             # V, offset tinh cua chip (datasheet V_OE: -40..+40 mV)
    sigma_hall_2k=3.5e-3,  # V rms trong bang 2 kHz (datasheet: 21 mV p-p ~ 6 sigma, CF=47 nF)
    # Troi nhiet tu phat nhiet (gia thiet, can bien luan trong M1):
    # dT_ss = kP * i^2 [degC], dV_th = TC * dT, quan tinh nhiet tau_th
    kP=1.5, TC=0.25e-3, tau_th=30.0,
)
FILTER = dict(
    RF=1.7e3, CF=100e-9,   # R_F(INT)=1.7 kOhm (datasheet), C_F chon 100 nF -> f_c1 = 936 Hz
    f_sk=600.0, Q_sk=0.7071,   # Sallen-Key Butterworth bac 2 (chong chong pho)
    use_sallen_key=True,       # False -> thi nghiem "khong co AA" de thay alias SMPS
)
EMI = dict(f_smps=31.4e3, A=25e-3)   # V, gon nguon xung DC-DC (gia thiet 50 mV p-p)
ADC = dict(bits=12, Vref=5.0, sigma_adc_lsb=0.3)   # nhieu noi ADC ~0.3 LSB rms

# ---------------- Kich ban 3 che do ----------------
# MODE1 = dict(I_ref=np.arange(0.0, 4.01, 0.5), T_each=2.0)       # hieu chuan tinh
MODE1 = dict(seq=[(0.0, 10.0)] + [s for I in [0.5, 4.0, 1.0, 3.5, 1.5, 3.0, 2.0, 2.5]
                                   for s in [(I, 2.0), (0.0, 1.0)]])
MODE2 = dict(tauL0=0.08, T=20.0)                                  # tai on dinh
MODE3 = dict(T=30.0,                                              # tai doi + tu phat nhiet
             profile=[(0.0, 0.03), (6.0, 0.12), (14.0, 0.06), (20.0, 0.06), (26.0, 0.15)])
# profile: (t_bat_dau, tau_L) ; doan 20->26 s la doc tuyen tinh 0.06 -> 0.15 N.m
