# EE6423 – Uoc luong dong trung binh dong co DC (ACS712-05B) – code nen tang

Python 3, NumPy 2.5.3, SciPy 1.18.1, Matplotlib 3.11.2 (phien ban da chay thu).

| Lenh | Ket qua |
|---|---|
| `python sim_dc_acs712.py` | data/mode1.npz, mode2.npz, mode3.npz |
| `python m1_expected_psd.py` | PSD du kien (ly thuyet) so voi Welch; figures/fig_m1_expected_psd.png |
| `python m2_quicklook.py` | in so lieu hieu chuan/thong ke/baseline; figures/fig_m2_psd_acf.png, fig_m3_timeseries.png |
| `python proteus_io.py export 2 0.5` | proteus/mode2_uraw.txt cho generator FILE cua Proteus |
| `python proteus_io.py compare <file_proteus.dat> 2 0.5` | sai lech Proteus - Python (mV, LSB) |

Moi tham so nam trong `config.py`. `filters/kalman_fixed.py` la ban mau Kalman 6 trang thai (chua tinh chinh).
