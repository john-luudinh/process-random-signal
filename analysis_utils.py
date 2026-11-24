"""analysis_utils.py - Cong cu Buoi 5-6: trung binh + KTC hieu chinh tri nho, ACF, Welch, LS."""
import numpy as np
from scipy import signal


def acf_biased(x, maxlag):
    """R_hat[l] = 1/N sum z~[n] z~[n-l]  (tru trung binh, chia N) -> luon hop le (Buoi 5)."""
    z = x - x.mean()
    N = len(z)
    r = np.array([np.dot(z[l:], z[:N - l]) / N for l in range(maxlag + 1)])
    return r, r / r[0]


def mean_ci(x, max_l=None, conf=1.96):
    """Trung binh mau + KTC 95% hieu chinh tri nho: kappa = 1 + 2 sum rho_hat[l],
    cong den lag truoc lag dau tien rho_hat < 0 (quy tac Buoi 5), |l| <= N/4."""
    N = len(x)
    max_l = max_l or N // 4
    _, rho = acf_biased(x, min(max_l, N // 4))
    neg = np.where(rho[1:] <= 0)[0]
    L = neg[0] if len(neg) else len(rho) - 1
    kappa = 1 + 2 * rho[1:L + 1].sum()
    s = x.std()
    Neff = N / kappa
    half = conf * s / np.sqrt(Neff)
    return dict(mean=x.mean(), std=s, kappa=kappa, Neff=Neff, ci=half, half_naive=conf * s / np.sqrt(N), L=L)


def welch_psd(x, fs, L, overlap=0.5, window="hann"):
    """PSD hai phia theo rad/sample se la S(e^jw) = P_onesided * fs/(4*pi) ... o day tra ve
    PSD mot phia theo Hz (V^2/Hz) va tan so Hz; doi sang w = 2*pi*f/fs khi ve."""
    f, P = signal.welch(x - x.mean(), fs=fs, window=window, nperseg=L, noverlap=int(L * overlap),
                        detrend=False, scaling="density")
    return f, P


def ls_fit(H, z):
    """LS: theta = (H^T H)^-1 H^T z ; Cov = sigma^2 (H^T H)^-1 (gia thiet nhieu trang)."""
    theta, *_ = np.linalg.lstsq(H, z, rcond=None)
    r = z - H @ theta
    dof = len(z) - H.shape[1]
    s2 = r @ r / dof
    cov = s2 * np.linalg.inv(H.T @ H)
    return theta, cov, r


def yule_walker(x, p):
    r, _ = acf_biased(x, p)
    R = np.array([[r[abs(i - j)] for j in range(p)] for i in range(p)])
    a = np.linalg.solve(R, r[1:p + 1])
    s2 = r[0] - a @ r[1:p + 1]
    return a, s2


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
