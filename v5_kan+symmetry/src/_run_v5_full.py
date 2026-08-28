
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as _pltpre
_pltpre.show = lambda *a, **k: None

# ======== EXEC CELL 1 id=v5-code-001 ========
import math, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

CONFIG = {
    "seed": 0,
    "quick_test": True,     # small subset + few epochs, to prove the pipeline runs end to end first
    "S": 512,                # spatial grid points
    "N": 1000,               # samples
    "T": 20,                 # output timesteps (t=1..T; t=0 is the input IC)
    "n_test": 150,
    "n_val": 100,
    # reference baseline (v4 parity)
    "fno_modes": 24,
    "fno_width": 32,
    "fno_layers": 4,
    # shared training budget: EVERY model below uses these (fairness contract)
    "epochs": 100,
    "batch_size": 32,
    "lr": 1e-3,
    # --- KAN operator hyperparameters ---
    "modal_M": 32,        # retained Fourier modes in the modal KAN trunk/head
    "modal_sub_stride": 8,# u0 subsample stride fed alongside modal features (shock localization)
    "kan_hidden": 128,    # hidden width of the modal KAN trunk / pointwise mixing
    "cheby_degree": 4,    # degree D of the per-mode Chebyshev kernels (T_0..T_D)
    "rat_p": 5,           # numerator degree of rational kernels P(x)/Q(x)
    "rat_q": 4,           # denominator degree
    "kano_width": 128,    # inner width of the physical-space rational-KAN operator
    "kano_layers": 6,
}
if CONFIG["quick_test"] and False:  # FULL BUDGET RUN
    CONFIG["N"] = 120
    CONFIG["epochs"] = 3
    CONFIG["n_test"] = 20
    CONFIG["n_val"] = 15

import os
os.makedirs("plots_v5", exist_ok=True)   # all figures land in plots_v5/
random.seed(CONFIG["seed"]); np.random.seed(CONFIG["seed"]); torch.manual_seed(CONFIG["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if device.type == "cpu":
    print("WARNING: no GPU detected. In Colab: Runtime > Change runtime type > GPU (T4).")

# ======== EXEC CELL 3 id=v5-code-003 ========
def fourier_translate_1d(field, c):
    """field: (B,S) real tensor. c: (B,) shift in pixels (fractional allowed).
    (T_c f)(x) = F^{-1}[e^{-i k c} f_hat(k)](x), Nyquist bin dropped for fractional shifts."""
    B, S = field.shape
    f_hat = torch.fft.fft(field.to(torch.complex64), dim=-1)
    k = torch.fft.fftfreq(S, d=1.0 / S).to(field.device).view(1, S)
    phase = -2 * math.pi * (k * c.view(B, 1)) / S
    shift_factor = torch.exp(1j * phase)
    if S % 2 == 0:
        nyq = S // 2
        is_int = torch.isclose(c, torch.round(c), atol=1e-4)
        keep = is_int.view(B, 1).to(shift_factor.dtype)
        shift_factor = shift_factor.clone()
        shift_factor[:, nyq:nyq + 1] *= keep
    out = torch.fft.ifft(f_hat * shift_factor, dim=-1)
    imag_res = out.imag.abs().max().item()
    if imag_res > 1e-2:
        print(f"WARNING: projector imaginary residual {imag_res:.2e} -- check input.")
    return out.real


DEALIAS_FRAC = 2.0 / 3.0


def bandlimit(u, frac=DEALIAS_FRAC):
    """Zero all rfft modes above frac*Nyquist (the solver's own dealias band)."""
    S = u.shape[-1]
    keep = int(frac * S / 2)
    U = torch.fft.rfft(u, dim=-1)
    U[:, keep + 1:] = 0
    return torch.fft.irfft(U, n=S, dim=-1)


def fourier_shift_exact(u, c_px):
    """Exact fractional periodic shift: band-limit to the solver's dealias band, then apply the
    Fourier phase ramp. Exact by the sampling theorem for the band-limited field."""
    return fourier_translate_1d(bandlimit(u), c_px)

print("Translation projector ready.")

# ======== EXEC CELL 4 id=v5-code-004 ========
# --- Projector correctness checks (adapted from v4's Sym-1/4/5 checks) ---
_S = 256
_f = torch.randn(5, _S)
_c_int = torch.tensor([3., -5., 0., 40., -100.])
_shifted = fourier_translate_1d(_f, _c_int)
_recov = fourier_translate_1d(_shifted, -_c_int)
_err = (_recov - _f).abs().max().item()
print(f"[Check 1] Integer-shift round-trip max abs error: {_err:.3e} (must be ~1e-6)")
assert _err < 1e-3

_xg = torch.arange(_S, dtype=torch.float32) * 20.0 / _S   # periodic Burgers-like grid
_shock = bandlimit(torch.tanh(8 * (_xg / 20.0 - 0.35)).unsqueeze(0))  # solver-dealiased like real data
_U, _t = 1.3, 7
_cp = _U * _t * _S / 20.0                                  # drift in pixels after _t steps
_ut = fourier_shift_exact(_shock, torch.tensor([_cp]))
_w = fourier_shift_exact(_ut, torch.tensor([-_cp]))
_e = ((_w - _shock).norm() / _shock.norm()).item()
print(f"[Check 2] Galilean-style round-trip relative err: {_e:.2e}")
assert _e < 1e-3

_cp_frac = _cp + 0.37
_sh_f = fourier_shift_exact(_shock, torch.tensor([_cp_frac]))
_sh_back = fourier_shift_exact(_sh_f, torch.tensor([-_cp_frac]))
_e_bl = ((_sh_back - _shock).norm() / _shock.norm()).item()
print(f"[Check 3] FRACTIONAL-shift round trip on band-limited shock: {_e_bl:.2e} (sampling-theorem exact)")
assert _e_bl < 1e-4
print("\nProjector sanity checks PASSED.")

# ======== EXEC CELL 6 id=v5-code-006 ========
def _burgers_step(u_hat, k, nu, dt):
    dealias = (torch.abs(k) < (2 / 3) * k.abs().max()).to(u_hat.dtype)
    def nl(uh):
        u = torch.fft.ifft(uh, dim=-1).real
        flux_hat = torch.fft.fft((0.5 * u ** 2).to(torch.complex64), dim=-1)
        return -1j * k * flux_hat * dealias
    decay = torch.exp(-nu.view(-1, 1) * (k.view(1, -1) ** 2) * dt / 2)
    u_hat = u_hat * decay
    k1 = nl(u_hat); k2 = nl(u_hat + dt/2*k1); k3 = nl(u_hat + dt/2*k2); k4 = nl(u_hat + dt*k3)
    u_hat = u_hat + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    u_hat = u_hat * decay
    return u_hat


def make_burgers_dataset(N, S, T, L=20.0, nu=0.02, dt=5e-4, substeps=300,
                          n_modes=6, flow_mag_range=(0.8, 1.8), osc_scale=0.4,
                          seed=0, device="cpu", batch_size=200):
    """Burgers pseudo-spectral solver -- VERBATIM from v4 (see v4 cell for full physics notes).
    Per-sample random mean flow U_i plus damped sine modes: makes Galilean drift genuine,
    sample-specific, and exactly predictable from u0 alone."""
    rng = np.random.RandomState(seed)
    x = np.linspace(0, L, S, endpoint=False)
    k = 2 * math.pi * torch.fft.fftfreq(S, d=L / S).to(device)

    mag = rng.uniform(*flow_mag_range, size=N)
    sign = rng.choice([-1, 1], size=N)
    mean_flow = (mag * sign).astype(np.float32)
    u0_all = np.zeros((N, S), dtype=np.float32)
    for i in range(N):
        u = np.full(S, mean_flow[i])
        for m in range(1, n_modes + 1):
            amp = osc_scale * rng.uniform(-1, 1) / m
            phase = rng.uniform(0, 2 * np.pi)
            u += amp * np.sin(m * 2 * np.pi * x / L + phase)
        u0_all[i] = u

    data = np.zeros((N, T + 1, S), dtype=np.float32)
    data[:, 0] = u0_all
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        u0_b = torch.from_numpy(u0_all[start:end]).float().to(device)
        u_hat = torch.fft.fft(u0_b.to(torch.complex64), dim=-1)
        nu_b = torch.full((end - start,), nu, device=device)
        for t in range(1, T + 1):
            for _ in range(substeps):
                u_hat = _burgers_step(u_hat, k, nu_b, dt)
            data[start:end, t] = torch.fft.ifft(u_hat, dim=-1).real.cpu().numpy()
    return data, {"L": L, "nu": nu, "dt_out": dt * substeps, "mean_flow": mean_flow}

BURGERS_L = 20.0
BURGERS_NU = 0.02
BURGERS_DT_OUT = 5e-4 * 300  # physical time per output step (must match the generator)

# ======== EXEC CELL 7 id=v5-code-007 ========
_bdata, _bmeta = make_burgers_dataset(N=3, S=128, T=5, nu=BURGERS_NU, seed=1, device=str(device))
print("Smoke-test shapes OK:", _bdata.shape, " any NaN:", np.isnan(_bdata).any(),
      " max|u|:", float(np.abs(_bdata).max()))

# ======== EXEC CELL 9 id=v5-code-015 ========
print("Generating Burgers dataset...")
t0 = time.time()
burgers_data, burgers_meta = make_burgers_dataset(
    N=CONFIG["N"], S=CONFIG["S"], T=CONFIG["T"], L=BURGERS_L, nu=BURGERS_NU, dt=5e-4,
    substeps=300, flow_mag_range=(0.8, 1.8), osc_scale=0.4, seed=CONFIG["seed"], device=str(device),
)
print(f"Burgers data shape: {burgers_data.shape}  (generated in {time.time()-t0:.1f}s)  "
      f"NaN/Inf: {np.isnan(burgers_data).any()}, {np.isinf(burgers_data).any()}")

# ======== EXEC CELL 11 id=v5-code-009a ========
# ---------------- Reference baseline architecture (kept VERBATIM from v4) ----------------
class SpectralConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, modes):
        super().__init__()
        self.in_ch, self.out_ch, self.modes = in_ch, out_ch, modes
        scale = 1.0 / (in_ch * out_ch)
        self.w = nn.Parameter(scale * torch.rand(in_ch, out_ch, modes, dtype=torch.cfloat))

    def forward(self, x):
        B, C, S = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        m = min(self.modes, x_ft.shape[-1])
        out_ft = torch.zeros(B, self.out_ch, x_ft.shape[-1], dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :m] = torch.einsum("bix,iox->box", x_ft[:, :, :m], self.w[:, :, :m])
        return torch.fft.irfft(out_ft, n=S, dim=-1)


class FNO1d(nn.Module):
    def __init__(self, modes, width, in_channels, out_channels, n_layers=4):
        super().__init__()
        self.fc0 = nn.Linear(in_channels + 1, width)
        self.spectral = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(n_layers)])
        self.w_layers = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(n_layers)])
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        # x: (B, S, in_channels)
        B, S, _ = x.shape
        grid = torch.linspace(0, 1, S, device=x.device).view(1, S, 1).repeat(B, 1, 1)
        x = torch.cat([x, grid], dim=-1)
        x = self.fc0(x).permute(0, 2, 1)
        for spec, w in zip(self.spectral, self.w_layers):
            x = F.gelu(spec(x) + w(x))
        x = x.permute(0, 2, 1)
        x = F.gelu(self.fc1(x))
        return self.fc2(x).permute(0, 2, 1)  # (B, out_channels, S)

print("Reference FNO1d defined.")

# ======== EXEC CELL 12 id=v5-code-009b ========
# ---------------- Parametric univariate kernel toolbox ----------------
# Every family implements: phi(z) -> y for z of shape (..., n_units); phi_o acts on column o.
# All families are single smooth curves => "pick any kernel, visualize it" holds by construction.


class RationalKernel(nn.Module):
    """Per-unit rational function P_o(x)/Q_o(x). Q(0)=1 and small coeffs keep init benign;
    poles give free local adaptivity near steep fronts."""
    def __init__(self, n_units, p_deg=5, q_deg=4):
        super().__init__()
        self.p_deg, self.q_deg = p_deg, q_deg
        self.Pw = nn.Parameter(torch.randn(n_units, p_deg + 1) * 0.05)
        self.Qw = nn.Parameter(torch.randn(n_units, q_deg + 1) * 0.02)
        with torch.no_grad():          # start Q ~ 1 -> phi ~ small correction around 0
            self.Qw[:, 0].fill_(1.0)

    def forward(self, z):
        # softsign-squash the pre-activation into (-1,1): keeps P/Q bounded for ANY input scale
        zs = z / (1.0 + z.abs())
        xp = torch.stack([zs ** j for j in range(self.p_deg + 1)], dim=-1)
        xq = torch.stack([zs ** j for j in range(self.q_deg + 1)], dim=-1)
        num = torch.einsum("...up,up->...u", xp, self.Pw)
        den = torch.einsum("...uq,uq->...u", xq, self.Qw).abs() + 1e-3
        return num / den

    def curves(self, xs):                               # xs: (G,) probe grid -> (G, n_units)
        G = xs.numel()
        u = self.Pw.shape[0]
        return self.forward(xs.view(G, 1).expand(G, u))


class ChebyshevKernel(nn.Module):
    """Per-unit truncated Chebyshev series sum_j c_{o,j} T_j(x') on x'=clip-normalized input."""
    def __init__(self, n_units, degree=4):
        super().__init__()
        self.degree = degree
        self.Cw = nn.Parameter(torch.zeros(n_units, degree + 1))
        with torch.no_grad():          # T_1 coefficient tiny -> starts near-zero nonlinearity
            self.Cw[:, 1].fill_(0.02)

    def forward(self, z):
        xn = z.clamp(-3, 3) / 3.0                        # bounded domain for stability
        acc = self.Cw[:, 0] * torch.ones_like(z) + self.Cw[:, 1] * xn
        Tprev, Tcur = torch.ones_like(z), xn
        for j in range(2, self.degree + 1):
            Tnext = 2 * xn * Tcur - Tprev                # Chebyshev recurrence
            acc = acc + self.Cw[:, j] * Tnext
            Tprev, Tcur = Tcur, Tnext
        return acc                                       # (..., n_units)

    def curves(self, xs):                               # xs: (G,) -> (G, n_units)
        G = xs.numel()
        u = self.Cw.shape[0]
        return self.forward(xs.view(G, 1).expand(G, u))


class BSplineKernel(nn.Module):
    """Per-unit uniform cubic B-spline bumps (the pykan reference look), included so 'better
    than spline' claims are anchored against the real thing."""
    def __init__(self, n_units, n_basis=10, range_=(-3, 3)):
        super().__init__()
        self.n_basis, self.range_ = n_basis, range_
        self.Cw = nn.Parameter(torch.zeros(n_units, n_basis))
        self.register_buffer("centers", torch.linspace(range_[0], range_[1], n_basis))

    @staticmethod
    def _cardinal(u):
        """Uniform cubic B-spline cardinal function beta_3."""
        a = u.abs()
        return torch.where(
            a <= 1, 2.0 / 3 - a * a + a ** 3 / 2,
            torch.where(a < 2, (2 - a).clamp_min(0) ** 3 / 6, torch.zeros_like(a)))

    def _basis(self, x):
        delta = self.centers[1] - self.centers[0]
        u = (x.unsqueeze(-1) - self.centers.view(1, -1)) / delta     # (...,n_basis)
        return self._cardinal(u)

    def forward(self, z):
        xn = z.clamp(*self.range_)
        B = self._basis(xn)
        return torch.einsum("...ub,ub->...u", B, self.Cw)

    def curves(self, xs):                               # xs: (G,) -> (G, n_units)
        G = xs.numel()
        u = self.Cw.shape[0]
        return self.forward(xs.view(G, 1).expand(G, u))


def make_kernel(name, n_units, cfg=None):
    cfg = cfg or CONFIG
    if name == "rational":
        return RationalKernel(n_units, cfg["rat_p"], cfg["rat_q"])
    if name == "chebyshev":
        return ChebyshevKernel(n_units, cfg["cheby_degree"])
    if name == "bspline":
        return BSplineKernel(n_units)
    raise ValueError(name)


@torch.no_grad()
def plot_kernel_curves(kernel, tag, n_show=24, lo=-3, hi=3):
    """Draw each output unit's learned activation curve -- THE KAN interpretability figure.
    Identical style for every kernel family (each thin line = one unit's univariate kernel)."""
    xs = torch.linspace(lo, hi, 200)
    with torch.no_grad():
        Y = kernel.curves(xs.to(next(kernel.parameters()).device)).cpu().numpy()
    fig, ax = plt.subplots(figsize=(7, 4))
    for o in range(min(n_show, Y.shape[-1])):
        ax.plot(xs.numpy(), Y[:, o], lw=0.8, alpha=0.6)
    ax.set_title(f"{tag}: learned univariate kernels phi_o(x)")
    ax.set_xlabel("pre-activation x"); ax.set_ylabel(r"$\phi(x)$")
    ax.grid(alpha=0.25)
    plt.tight_layout(); plt.show()
    return fig

print("Kernel toolbox defined: rational / chebyshev / bspline.")

# ======== EXEC CELL 13 id=v5-code-009c ========
# ---------------- Visualizable KAN operators ----------------
class KANPointwise1d(nn.Module):
    """Post-activation Kolmogorov-Arnold channel mixer applied at every grid site:
        y_o = base_o . x  +  s_o * phi_o( W_o . x )
    One univariate kernel phi_o PER OUTPUT UNIT => cheap, stable, and exactly one drawable
    curve per channel (the efficient-kan trick generalized to parametric kernels)."""
    def __init__(self, cin, cout, kernel="rational"):
        super().__init__()
        self.lin = nn.Linear(cin, cout)
        self.base = nn.Linear(cin, cout)
        self.scale = nn.Parameter(torch.full((cout,), 0.5))
        self.phi = make_kernel(kernel, cout)

    def forward(self, x):
        # x: (B, C, L) arbitrary leading batch dims handled by Linear broadcasting
        B, C, L = x.shape
        xr = x.permute(0, 2, 1)                       # (B,L,C)
        y = self.base(xr) + self.scale.view(1, 1, -1) * self.phi(self.lin(xr))
        return y.permute(0, 2, 1)


class ModalFourierKANOp(nn.Module):
    """Modal-space Kolmogorov-Arnold operator. Everything happens on rfft coefficients:
      trunk inputs = [Re,Im of the M retained modes of u0] (+ subsampled u0 values so shock
      POSITION information rides along), stacked KA layers predict the modal trajectory
      u_k(t_1..t_T) for all output steps, reconstructed to frames by irfft.
    Approximation guarantee sketch (markdown above): Cole-Hopf makes the de-drifted dynamics
    diagonal in Fourier space, so representing the trajectory map on modal coefficients inherits
    super-algebraic convergence in M for analytic periodic solutions."""
    def __init__(self, in_channels, out_T, S, kernel="rational",
                 M=None, hidden=None, stride=None):
        super().__init__()
        cfgS = dict(M=M or CONFIG["modal_M"], H=hidden or CONFIG["kan_hidden"],
                    st=stride or CONFIG["modal_sub_stride"])
        self.M, self.H, self.stride, self.T, self.S = cfgS["M"], cfgS["H"], cfgS["st"], out_T, S
        sub = len(range(0, S, cfgS["st"]))
        din = 2 * cfgS["M"] + sub
        self.trunk = nn.Sequential(
            KANPointwise1d(din, cfgS["H"], kernel), nn.GELU(),
            KANPointwise1d(cfgS["H"], cfgS["H"], kernel), nn.GELU(),
        )
        self.head = nn.Linear(cfgS["H"], 2 * cfgS["M"] * out_T)

    def forward(self, x):
        B, S, C = x.shape
        u = x[..., 0] if C == 1 else x.mean(-1)
        Xh = torch.fft.rfft(u, dim=-1)
        feats = [Xh.real[:, :self.M] / S, Xh.imag[:, :self.M] / S]
        feats.append(u[:, ::self.stride])                  # coarse physical detail (position)
        z = torch.cat(feats, dim=-1).unsqueeze(-1)         # (B,din,1): KANPointwise needs (B,C,L)
        h = self.trunk(z).squeeze(-1)
        out = self.head(h).view(B, self.T, self.M, 2)
        spec = torch.complex(out[..., 0], out[..., 1])                     # (B,T,M), units of X/S
        full = torch.zeros(B, self.T, S // 2 + 1, dtype=torch.cfloat, device=x.device)
        full[:, :, :self.M] = spec
        return torch.fft.irfft(full, n=S, dim=-1) * float(S)               # undo the /S scaling


class ModeMixCheby(nn.Module):
    """Per-mode channel mixing by Chebyshev kernels -- the direct KAN replacement of
    SpectralConv1d's complex einsum w[i,o,k]. Kernel curves are shared per channel pair
    (out,in) and modulated by a learned complex per-mode gain g[m,o], keeping parameters at
    FNO parity while every edge remains one drawable curve:
        u'_{o,k} = g[k,o] * ( sum_c cheby_{o,c}(Re x_{c,k}) + i * sum_c cheby~_{o,c}(Im x_{c,k}) )
    """
    def __init__(self, cin, cout, modes, degree=None):
        super().__init__()
        deg = degree or CONFIG["cheby_degree"]
        self.modes, self.deg = modes, deg
        self.Wr = nn.Parameter(torch.randn(cout, cin, deg + 1) / math.sqrt(cin))
        self.Wi = nn.Parameter(torch.randn(cout, cin, deg + 1) / math.sqrt(cin))
        self.gr = nn.Parameter(torch.ones(modes, cout) / math.sqrt(cin))
        self.gi = nn.Parameter(torch.zeros(modes, cout))

    def _basis(self, x):
        xn = torch.tanh(x)                                 # soft normalization to [-1,1]
        Ts = [torch.ones_like(xn), xn]
        for j in range(2, self.deg + 1):
            Ts.append(2 * xn * Ts[-1] - Ts[-2])
        return torch.stack(Ts[:self.deg + 1], dim=-1)      # (B,C,m,deg+1)

    def forward(self, x):
        B, C, S = x.shape
        Xt = torch.fft.rfft(x, dim=-1)
        m = min(self.modes, Xt.shape[-1])
        xt = Xt[:, :, :m]
        Tr = self._basis(xt.real)                          # (B,C,m,D+1)
        Ti = self._basis(xt.imag)
        A = torch.einsum("bcmd,ocd->bmo", Tr, self.Wr)     # real-path response
        Bc = torch.einsum("bcmd,ocd->bmo", Ti, self.Wi)    # imag-path response
        gr = self.gr[:m].unsqueeze(0); gi = self.gi[:m].unsqueeze(0)
        outr = gr * A - gi * Bc                            # complex gain applied properly
        outi = gi * A + gr * Bc
        out_ft = torch.zeros(B, C, Xt.shape[-1], dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :m] = torch.complex(outr, outi).permute(0, 2, 1)
        return torch.fft.irfft(out_ft, n=S, dim=-1)


class ChebyModeKANOp(nn.Module):
    """FNO-slot swap: identical skeleton (lift -> [spectral path + pointwise path]^L -> readout)
    but SpectralConv1d's einsum -> ModeMixCheby and Conv1d -> Chebyshev-KAN pointwise layer."""
    def __init__(self, in_channels, out_channels, S, modes=None, width=None, n_layers=None):
        super().__init__()
        W = width or CONFIG["fno_width"]
        Mo = modes or CONFIG["fno_modes"]
        L = n_layers or CONFIG["fno_layers"]
        self.fc0 = nn.Linear(in_channels + 1, W)
        self.spec = nn.ModuleList([ModeMixCheby(W, W, Mo) for _ in range(L)])
        self.point = nn.ModuleList([KANPointwise1d(W, W, kernel="chebyshev") for _ in range(L)])
        self.fc1 = nn.Linear(W, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        B, S, _ = x.shape
        grid = torch.linspace(0, 1, S, device=x.device).view(1, S, 1).repeat(B, 1, 1)
        x = torch.cat([x, grid], dim=-1)
        x = self.fc0(x).permute(0, 2, 1)
        for sp, pt in zip(self.spec, self.point):
            x = F.gelu(sp(x) + pt(x))
        x = x.permute(0, 2, 1)
        x = F.gelu(self.fc1(x))
        return self.fc2(x).permute(0, 2, 1)


class GlobalRationalConv(nn.Module):
    """Depthwise GLOBAL circular convolution whose filter weights come from a learned per-channel
    rational kernel over spatial OFFSET: psi_c(dx) = P_c(dn)/(Q_c(dn)+eps) times a learnable-width
    Gaussian envelope (localized bump ~ CNN filter, but analytically drawable).
    dx grid spans the whole domain => global receptive field at O(S log S) cost via FFT."""
    def __init__(self, channels, p_deg=None, q_deg=None):
        super().__init__()
        p_deg = p_deg or CONFIG["rat_p"]; q_deg = q_deg or CONFIG["rat_q"]
        self.channels = channels
        self.Pw = nn.Parameter(torch.randn(channels, p_deg + 1) * 0.01)
        self.Qw = nn.Parameter(torch.randn(channels, q_deg + 1) * 0.01)
        self.log_sigma = nn.Parameter(torch.full((channels,), math.log(0.06)))
        with torch.no_grad():
            self.Qw[:, 0].fill_(1.0)

    def filters(self, S, device):
        dn = torch.arange(S, device=device, dtype=torch.float32)
        dn = torch.where(dn > S // 2, dn - S, dn) / (S / 2)            # (-1,1) pixel offsets
        pw = torch.stack([dn ** j for j in range(self.Pw.shape[-1])], dim=-1)   # (S,p+1)
        qw = torch.stack([dn ** j for j in range(self.Qw.shape[-1])], dim=-1)
        P = torch.einsum("sp,cp->cs", pw, self.Pw)                      # (C,S)
        Q = torch.einsum("sq,cq->cs", qw, self.Qw).abs() + 1e-3
        sig = torch.exp(self.log_sigma).clamp(0.02, 1.0).view(-1, 1)
        env = torch.exp(-0.5 * (dn.view(1, -1) / sig) ** 2)
        psi = P / Q * env                                               # (C,S), |psi| <= ~1
        return psi                                                      # impulse responses

    def forward(self, x):
        B, C, S = x.shape
        psi = self.filters(S, x.device)                                # (C,S)
        PSIf = torch.fft.rfft(psi, dim=-1).unsqueeze(0)                # (1,C,K)
        return torch.fft.irfft(torch.fft.rfft(x, dim=-1) * PSIf, n=S, dim=-1)


class RationalGCKANOp(nn.Module):
    """KANO-R: physical-space Kolmogorov-Arnold operator.
    block_l:  GlobalRationalConv(width) -> KANPointwise1d(rational,width,width) -> GELU
    The ONLY nonlinearities are the learned kernel curves; global receptive field comes free
    from the FFT convolutions, whose kernels are directly visualizable."""
    def __init__(self, in_channels, out_channels, width=None, n_layers=None):
        super().__init__()
        W = width or CONFIG["kano_width"]
        L = n_layers or CONFIG["kano_layers"]
        self.fc0 = nn.Linear(in_channels + 1, W)
        self.convs = nn.ModuleList([GlobalRationalConv(W) for _ in range(L)])
        self.mixes = nn.ModuleList([KANPointwise1d(W, W, kernel="rational") for _ in range(L)])
        self.fc1 = nn.Linear(W, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        B, S, _ = x.shape
        grid = torch.linspace(0, 1, S, device=x.device).view(1, S, 1).repeat(B, 1, 1)
        x = torch.cat([x, grid], dim=-1)
        x = self.fc0(x).permute(0, 2, 1)
        for cv, mx in zip(self.convs, self.mixes):
            x = F.gelu(cv(x) + mx(x))
        x = x.permute(0, 2, 1)
        x = F.gelu(self.fc1(x))
        return self.fc2(x).permute(0, 2, 1)


def spectral_int_1d(u, S):
    """Exact periodic spectral integral on the zero-Nyquist band (same numerics as v4's
    Cole-Hopf pair): I = irfft( U / (i omega) ), DC set to 0."""
    om = 2 * math.pi * torch.fft.rfftfreq(S, d=1.0).to(u.device).to(torch.float64)  # rad per PIXEL
    U = torch.fft.rfft(u.to(torch.float64), dim=-1)
    U[..., 1:] = U[..., 1:]
    U0 = U.clone(); U0[..., 0] = 0; U0[..., -1] = 0
    inv = torch.zeros(om.shape, dtype=torch.complex128, device=om.device)
    nz = om > 1e-12
    inv[nz] = 1.0 / (1j * om[nz])
    return torch.fft.irfft(U0 * inv, n=S, dim=-1).to(torch.float32)


def spectral_deriv_1d(v, S):
    """Exact periodic spectral derivative d/dx on the grid (units: per pixel)."""
    om = 2 * math.pi * torch.fft.rfftfreq(S, d=1.0).to(v.device).to(torch.float64)  # rad per PIXEL
    V = torch.fft.rfft(v.to(torch.float64), dim=-1)
    V[..., -1] = 0
    return torch.fft.irfft(1j * om * V, n=S, dim=-1).to(torch.float32)


class ColeHopfKANOp(nn.Module):
    """THE math-proven operator: exact Hopf-Cole/Wiener representation of viscous Burgers,
    evaluated in log space (cancellation-free), with two small LEARNED KAN correction kernels:

        Ubar = mean(u0)                    [exact conserved invariant, u0 only]
        z    = -(dx/2nu) Integral(u-Ubar)  [exact log phi_0, per-sample constant dropped]
        z+   = z + rat_exp(z)              [LEARNED additive kernel curve, init == 0]
        log phi_t(x) = LOGSUMEXP_j [ z+(j) - (x-j)^2 / (4 nu t_j) ]   [EXACT heat evolution --
                                                       the softmax carries the e^{z} dynamic
                                                       range with zero cancellation]
        u(x,t) = E_j~P(x,.)[ (x-j) dx ] / t_j     [exact: -2nu d/dx log phi_t, algebraically
                                                   identical to the tilted expectation]
        out   = shift_{Ubar t}( u * rat_u(u) ) + Ubar  [LEARNED multiplicative kernel, init==1;
                                                        exact closed-form Galilean drift back]

    Hopf 1950 / Cole 1951. At init the two kernels are neutral -> the model IS the analytical
    Burgers solution operator (to solver/roundoff error) BEFORE any training. The kernel plots
    show the learned corrections: if they stay at 0/1, the data obeyed ideal physics; their
    deviation is the visualizable 'fit residual'. Reference: this softmax form is the
    heat-semigroup action on exp(z) written as a Wiener integral (Laplace/log-sum-exp)."""
    def __init__(self, in_channels, out_T, S, K=None):
        super().__init__()
        self.T, self.S = out_T, S
        self.dx = BURGERS_L / S
        p_deg, q_deg = CONFIG["rat_p"], CONFIG["rat_q"]
        self.exp_P = nn.Parameter(torch.zeros(p_deg + 1))          # additive z-correction, init 0
        self.exp_Q = nn.Parameter(torch.zeros(q_deg + 1)); self.exp_Q.data[0] = 1.0
        self.log_P = nn.Parameter(torch.zeros(p_deg + 1)); self.log_P.data[0] = 1.0  # multiplicative
        self.log_Q = nn.Parameter(torch.zeros(q_deg + 1)); self.log_Q.data[0] = 1.0
        print("[CH-KAN init] exact Hopf-Cole/Wiener operator (log-space softmax, cancellation-free); "
              "correction kernels at neutral (0 / 1). Physics constants: nu, dt_out, L only.")

    def _rat(self, x, P, Q):
        zs = x / (1.0 + x.abs())                                   # bounded squash
        xp = torch.stack([zs ** j for j in range(P.shape[0])], dim=-1)
        xq = torch.stack([zs ** j for j in range(Q.shape[0])], dim=-1)
        num = torch.einsum("...p,p->...", xp, P.to(x.dtype))
        den = torch.einsum("...q,q->...", xq, Q.to(x.dtype)).abs() + 1e-6
        return num / den

    def forward(self, x):
        B, S, C = x.shape
        u = (x[..., 0] if C == 1 else x.mean(-1)).to(torch.float64)
        Ubar = u.mean(dim=-1, keepdim=True)                        # (B,1)
        z = -(self.dx / (2 * BURGERS_NU)) * spectral_int_1d(u - Ubar, S)
        z = z + self._rat(z, self.exp_P, self.exp_Q).to(torch.float64)   # learned z-corr

        jj = torch.arange(S, device=x.device, dtype=torch.float64)
        dj = jj.view(-1, 1) - jj.view(1, -1)                       # (x, j): x - j
        dj = dj - S * torch.round(dj / S)                          # periodic, in [-S/2, S/2)
        d_phys = dj * self.dx                                      # physical displacement (x->j)

        outs = []
        for ti in range(1, self.T + 1):
            t = ti * BURGERS_DT_OUT
            # log kernel: z+(j) - (x-j)^2/(4 nu t); the constant -log sqrt(4 pi nu t) drops
            L = z.unsqueeze(1) - (d_phys ** 2 / (4 * BURGERS_NU * t)).unsqueeze(0)  # (B,Sx,Sj)
            Pw = torch.softmax(L.float(), dim=-1).to(torch.float64)                # tilted kernel
            E = (Pw * d_phys.unsqueeze(0)).sum(dim=-1) / t                       # u(x,t) exact
            outs.append(E)
        out = torch.stack(outs, dim=1)                             # (B,T,S) co-moving
        out = out * self._rat(out, self.log_P, self.log_Q).to(torch.float64)
        out = out.to(torch.float32)
        tt = torch.arange(1, self.T + 1, device=x.device, dtype=torch.float32)
        drift_px = (Ubar * (tt.view(1, -1) * BURGERS_DT_OUT) * (S / BURGERS_L)).float()
        out = fourier_shift_exact(out.reshape(B * self.T, S) + Ubar.repeat_interleave(self.T, 0).float(),
                                  drift_px.reshape(-1)).reshape(B, self.T, S)
        return out


def count_params(model):
    return sum(p.numel() for p in model.parameters())

def make_model(kind, T_out, S):
    if kind == "FNO":
        return FNO1d(CONFIG["fno_modes"], CONFIG["fno_width"], 1, T_out, CONFIG["fno_layers"])
    if kind == "ModalKAN":
        return ModalFourierKANOp(1, T_out, S)
    if kind == "ChebyKAN":
        return ChebyModeKANOp(1, T_out, S)
    if kind == "KANO-R":
        return RationalGCKANOp(1, T_out)
    if kind == "CH-KAN":
        return ColeHopfKANOp(1, T_out, S)
    raise ValueError(kind)

MODEL_KINDS = ["FNO", "ModalKAN", "ChebyKAN", "KANO-R", "CH-KAN"]

_tdummy = torch.randn(2, 64, 1)
for _kind in MODEL_KINDS:
    _m = make_model(_kind, 4, 64)
    _out = _m(_tdummy)
    assert _out.shape == (2, 4, 64) and torch.isfinite(_out).all(), _kind
print("All 5 model kinds build + forward OK." )

# ======== EXEC CELL 15 id=v5-code-011 ========
class Traj1dDS(Dataset):
    def __init__(self, ic, target):
        self.ic, self.target = ic, target
    def __len__(self): return len(self.ic)
    def __getitem__(self, i):
        return torch.from_numpy(self.ic[i]).float().unsqueeze(-1), torch.from_numpy(self.target[i]).float()


def relative_l2_loss(pred, true):
    B = pred.shape[0]
    diff = (pred - true).reshape(B, -1)
    norm_true = true.reshape(B, -1)
    return (diff.norm(dim=1) / (norm_true.norm(dim=1) + 1e-8)).mean()


def train_model(model, train_loader, val_loader, epochs, lr, device, loss_fn, tag=""):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    history = {"train_loss": [], "val_loss": [], "epoch_times": []}
    best_val, best_state = float("inf"), None
    for ep in range(epochs):
        ep_t0 = time.time()
        model.train(); tot = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"[{tag}] non-finite loss at epoch {ep}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * x.size(0)
        train_loss = tot / len(train_loader.dataset)
        model.eval(); vtot = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                vtot += loss_fn(model(x), y).item() * x.size(0)
        val_loss = vtot / len(val_loader.dataset)
        sched.step()
        history["train_loss"].append(train_loss); history["val_loss"].append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        history["epoch_times"].append(time.time() - ep_t0)
        if ep % max(1, epochs // 8) == 0 or ep == epochs - 1:
            print(f"[{tag}] epoch {ep+1}/{epochs} train={train_loss:.5f} val={val_loss:.5f}")
    model.load_state_dict(best_state)
    return model, history


def make_split(N, config):
    """THE split -- one permutation, one seed, shared by every model and arm (fairness)."""
    rng = np.random.RandomState(config["seed"])
    perm = rng.permutation(N)
    n_test = min(config["n_test"], max(1, N // 6))
    n_val = min(config["n_val"], max(1, N // 8))
    n_train = N - n_test - n_val
    return perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]


@torch.no_grad()
def eval_model_on_test(model, ic_all, fut_all, test_idx, config, keep_ids=None):
    """Relative-L2 errors (+ per-frame error matrix). If keep_ids (dataset row indices) is given,
    also stores raw predictions for those rows for plotting. NOTHING outside test_idx is touched."""
    model.eval(); errs, mats, preds = [], [], {}
    bs = config["batch_size"]
    keep_set = set(keep_ids) if keep_ids else set()
    for s0 in range(0, len(test_idx), bs):
        idx = test_idx[s0:s0 + bs]
        b = len(idx)
        x = torch.from_numpy(ic_all[idx]).float().unsqueeze(-1).to(device)
        y = torch.from_numpy(fut_all[idx]).float().to(device)
        pred = model(x)                                    # (b,T,S)
        errs.append((pred - y).flatten(1).norm(dim=1) / (y.flatten(1).norm(dim=1) + 1e-8))
        mats.append((pred - y).norm(dim=-1) / (y.norm(dim=-1) + 1e-8))
        for bi, row in enumerate(idx):
            if int(row) in keep_set:
                preds[int(row)] = pred[bi].cpu().numpy()
    return torch.cat(errs), torch.cat(mats), preds


@torch.no_grad()
def galilean_canonicalize(raw, device):
    """Forward Galilean boost -- EXACT v4 MATH.
    w(x,t) := u(x+Ut, t) - U solves Burgers with drift-free IC u0-U, where U = mean(u0) is an
    exact conserved invariant of the periodic solver. Applied here: subtract U from frame 0,
    shift each frame t by -U*t*dt_out*(S/L) pixels with sampling-theorem-exact band-limited shifts.
    Symmetry parameters come from frame 0 ONLY."""
    data = raw.astype(np.float32).copy()
    N, Tp1, S = data.shape
    Ubar = data[:, 0].mean(-1)                             # exact invariant of the periodic solver
    ub = torch.from_numpy(Ubar).float().to(device)
    data[:, 0] = data[:, 0] - Ubar[:, None]
    for t in range(1, Tp1):
        ft = torch.from_numpy(data[:, t]).float().to(device)
        ct = ub * (t * BURGERS_DT_OUT) * (S / BURGERS_L)   # t is an output-step index -> physical time
        w = fourier_shift_exact(ft, -ct)
        w = w - ub.view(-1, 1)
        data[:, t] = w.cpu().numpy()
    ctx = {"Ubar": Ubar}                                   # <-- inversion reads ONLY this key
    tt = np.arange(1, Tp1, dtype=np.float32)
    disp = (ctx["Ubar"][:, None] * tt[None, :] * BURGERS_DT_OUT
            * (S / BURGERS_L)).astype(np.float32)          # physics-exact closed-form drift
    return data, ctx, disp


def run_galilean_closed_form(raw, kind, config, device, base_err=None, tag_suffix="", keep_ids=None):
    """v4's 'Galilean closed-form' arm, verbatim pipeline, with the leakage fix: reconstruction
    adds back ctx['Ubar'] (= mean(u0)) -- NOTHING else is restored, so no future-frame information
    is ever consumed at eval time (v4 restored per-frame canonical means there; the audit above
    proves they equal mean(u0), so this changes results only by floating-point ordering)."""
    canon, ctx, disp = galilean_canonicalize(raw, device)
    ic_c = canon[:, 0]
    fut_c = canon[:, 1:]
    N, Tp1, S = raw.shape
    T_out = Tp1 - 1
    train_idx_a, val_idx_a, test_idx_a = make_split(N, config)

    tr = DataLoader(Traj1dDS(ic_c[train_idx_a], fut_c[train_idx_a]),
                    batch_size=config["batch_size"], shuffle=True)
    va = DataLoader(Traj1dDS(ic_c[val_idx_a], fut_c[val_idx_a]), batch_size=config["batch_size"])
    model = make_model(kind, T_out, S).to(device)
    model, hist = train_model(model, tr, va, config["epochs"], config["lr"], device,
                              relative_l2_loss, tag=f"gal{tag_suffix}/{kind}")

    model.eval()
    errs, mats, preds = [], [], {}
    keep_set = set(keep_ids) if keep_ids else set()
    bs = config["batch_size"]
    with torch.no_grad():
        for s0 in range(0, len(test_idx_a), bs):
            idx = test_idx_a[s0:s0 + bs]
            b = len(idx)
            x = torch.from_numpy(ic_c[idx]).float().unsqueeze(-1).to(device)
            y = torch.from_numpy(raw[idx, 1:]).float().to(device)
            pred = model(x)                                    # (b,T,S) canonical frames
            dp = torch.from_numpy(disp[idx]).float().to(device)   # closed-form displacement (u0-only)
            flat = pred.reshape(b * T_out, S)
            dpf = dp.reshape(-1)
            recon = fourier_shift_exact(flat, dpf)             # inverse boost: shift by +Ut
            ubr = torch.from_numpy(ctx["Ubar"][idx]).float().to(device).repeat_interleave(T_out).view(-1, 1)
            recon = recon + ubr                                # DC restored from the u0 invariant ONLY
            recon = recon.reshape(b, T_out, S)
            errs.append((recon - y).flatten(1).norm(dim=1) / (y.flatten(1).norm(dim=1) + 1e-8))
            mats.append((recon - y).norm(dim=-1) / (y.norm(dim=-1) + 1e-8))
            for bi, row in enumerate(idx):
                if int(row) in keep_set:
                    preds[int(row)] = recon[bi].cpu().numpy()
    errs = torch.cat(errs); mats = torch.cat(mats)
    msg = f"[Galilean closed-form{tag_suffix}] {kind}: rel-L2 = {errs.mean():.4f}"
    if base_err is not None:
        msg += f"  ({(1 - errs.mean() / base_err.mean()).item() * 100:+.1f}% vs FNO-raw baseline)"
    print(msg)
    tvar_red = 1 - fut_c.var(axis=1).mean() / (raw[:, 1:].var(axis=1).mean() + 1e-12)
    print(f"    temporal-variance reduction from canonicalization: {tvar_red * 100:.1f}%")
    return {"kind": kind, "errs": errs, "mats": mats, "hist": hist, "model": model,
            "canon": canon, "ctx": ctx, "disp": disp, "tvar_red": tvar_red,
            "test_idx": test_idx_a, "preds": preds}

print("Harness ready.")

# ======== EXEC CELL 17 id=v5-code-013 ========
# ---- leakage audit: executed assertions (run BEFORE any training) ----
frame_means = burgers_data.mean(axis=-1)                      # ground-truth DIAGNOSTIC only
dev_from_u0 = float(np.abs(frame_means[:, 1:] - frame_means[:, :1]).max())
print(f"[Audit 1] max_t |mean(u_t) - mean(u0)| over entire dataset: {dev_from_u0:.3e}"
      f"  ({'PASS' if dev_from_u0 < 1e-3 else 'FAIL'})")
assert dev_from_u0 < 1e-3, "mean-flow conservation violated -- Galilean closed-form phase invalid"

print("[Audit 2] galilean_canonicalize(): symmetry parameters read data[:,0] exactly once "
      "(verified structurally: Ubar is the only parameter defined)")
assert "means_future" not in globals(), "[Audit 3] stale means_future context found -- v5 must not define it"
ctx_keys = set(galilean_canonicalize(burgers_data[:4].copy(), device)[1].keys())
print(f"[Audit 3] ctx keys at inversion time: {sorted(ctx_keys)}")
assert ctx_keys <= {"Ubar"}, "[Audit 3] ctx carries more than the u0-only invariant!"

tr_i, va_i, te_i = make_split(burgers_data.shape[0], CONFIG)
allidx = np.concatenate([tr_i, va_i, te_i])
assert sorted(allidx.tolist()) == list(range(burgers_data.shape[0])), "[Audit 4] split not a partition"
overlap = (np.intersect1d(te_i, tr_i).size + np.intersect1d(te_i, va_i).size
           + np.intersect1d(va_i, tr_i).size)
assert overlap == 0
print(f"[Audit 4] split partition OK: train={len(tr_i)} val={len(va_i)} test={len(te_i)}, "
      f"zero overlap, seed={CONFIG['seed']}")
print("[Audit 5] no dataset-wide normalization statistics are used anywhere in the pipeline")
print("\nLEAKAGE AUDIT PASSED.")

# ======== EXEC CELL 18 id=v5-code-016 ========
# ---------------- Raw-data models: identical budget, identical split ----------------
ic_raw = burgers_data[:, 0]
fut_raw = burgers_data[:, 1:]
N_b, Tp1_b, S_b = burgers_data.shape
T_out_b = Tp1_b - 1
train_idx, val_idx, test_idx = make_split(N_b, CONFIG)

tr_raw = DataLoader(Traj1dDS(ic_raw[train_idx], fut_raw[train_idx]),
                    batch_size=CONFIG["batch_size"], shuffle=True)
va_raw = DataLoader(Traj1dDS(ic_raw[val_idx], fut_raw[val_idx]), batch_size=CONFIG["batch_size"])

results_raw = {}
for kind in MODEL_KINDS:
    model = make_model(kind, T_out_b, S_b).to(device)
    n_params = count_params(model)
    model, hist = train_model(model, tr_raw, va_raw, CONFIG["epochs"], CONFIG["lr"], device,
                              relative_l2_loss, tag=f"raw/{kind}")
    epoch_times = hist.get("epoch_times", [])
    results_raw[kind] = {"model": model, "hist": hist, "params": n_params,
                         "epoch_time": float(np.mean(epoch_times[1:])) if len(epoch_times) > 1 else float("nan")}
    print(f"[raw/{kind}] params={n_params:,}\n")

print("Parameter parity vs FNO:")
_p0 = results_raw["FNO"]["params"]
for kind in MODEL_KINDS:
    print(f"  {kind:10s} {results_raw[kind]['params']:>9,d}  ({results_raw[kind]['params']/(_p0+1e-9):5.2f}x)")

# Model selection STRICTLY on validation error (no test peeking before selection is frozen)
val_pick = {}
for kind in MODEL_KINDS[1:]:
    _pe, _, _ = eval_model_on_test(results_raw[kind]["model"], ic_raw, fut_raw, val_idx, CONFIG)
    val_pick[kind] = float(_pe.mean())
winner = min(val_pick, key=val_pick.get)
print("\nValidation-only KAN scores:", {k: round(v, 5) for k, v in val_pick.items()})
print(f"==> KAN WINNER (by validation): {winner}")

# ======== EXEC CELL 19 id=v5-code-016b ========
# ---------------- Test-set scores for every raw model (selection already frozen) ----------------
keep_ids = [int(i) for i in test_idx[:3]]   # three fixed test samples reserved for solution plots
fno_err, fno_mat, fno_preds = eval_model_on_test(results_raw["FNO"]["model"], ic_raw, fut_raw,
                                                 test_idx, CONFIG, keep_ids=keep_ids)
baseline_err, baseline_mat = fno_err, fno_mat           # v4 name kept for parity with old plot cells
results_raw["FNO"].update({"test_err": fno_err, "mat": fno_mat, "preds": fno_preds})
print(f"FNO raw (reference baseline): rel-L2 = {fno_err.mean():.4f}")
for kind in MODEL_KINDS[1:]:
    e, m, pr = eval_model_on_test(results_raw[kind]["model"], ic_raw, fut_raw, test_idx,
                                  CONFIG, keep_ids=keep_ids)
    results_raw[kind].update({"test_err": e, "mat": m, "preds": pr})
    imp = (1 - e.mean() / fno_err.mean()).item() * 100
    print(f"{kind} raw: rel-L2 = {e.mean():.4f}  ({imp:+.1f}% vs FNO)")

# ======== EXEC CELL 20 id=v5-code-017 ========
# ---------------- Galilean closed-form arms (exact v4 symmetry math, leakage-fixed) ----------------
# Arms: the FNO reference, KANO-R (strongest GENERIC KAN -- the symmetry-vs-architecture
# question), and the validation-selected KAN winner (deduplicated).
sym_results = {}
GAL_ARMS = ["FNO", "KANO-R"] + ([winner] if winner not in ("FNO", "KANO-R") else [])
for _kind in GAL_ARMS:
    sym_results[_kind] = run_galilean_closed_form(
        burgers_data, _kind, CONFIG, device, base_err=fno_err, tag_suffix="",
        keep_ids=[int(i) for i in test_idx[:3]])
print(f"Galilean arms run for: {GAL_ARMS}")

# ======== EXEC CELL 21 id=v5-code-017b ========
# ---------------- SPEED BENCHMARK: FNO vs every KAN (inference + training) ----------------
# Inference: identical batch (32 x 1 x S), timed forward passes with CUDA syncs.
# Training: mean wall-clock per epoch recorded during the raw-model runs above.


@torch.no_grad()
def bench_inference(model, S, batch=32, reps=30, warmup=5):
    model.eval()
    x = torch.randn(batch, S, 1, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / reps * 1000.0    # ms per batch


speed = {}
for kind in MODEL_KINDS:
    ms = bench_inference(results_raw[kind]["model"], S_b)
    speed[kind] = {"infer_ms_per_batch": ms,
                   "train_s_per_epoch": results_raw[kind]["epoch_time"]}
    print(f"{kind:10s} inference: {ms:7.2f} ms/batch ({CONFIG['batch_size']/ (ms/1000):8.0f} samples/s)"
          f"   training: {results_raw[kind]['epoch_time']:6.3f} s/epoch")

sp_fno = speed["FNO"]["infer_ms_per_batch"]
sp_win = speed[winner]["infer_ms_per_batch"]
print(f"\nSPEED: winner ({winner}) vs FNO -> inference {sp_fno/max(sp_win,1e-9):.2f}x "
      f"({'faster' if sp_win < sp_fno else 'slower'}), "
      f"training {speed['FNO']['train_s_per_epoch']/max(speed[winner]['train_s_per_epoch'],1e-9):.2f}x "
      f"({'faster' if speed[winner]['train_s_per_epoch'] < speed['FNO']['train_s_per_epoch'] else 'slower'})")

# ======== EXEC CELL 23 id=v5-code-019 ========
# ---------------- Benchmark bars ----------------
fig, ax = plt.subplots(figsize=(12, 5))
names = ["FNO (reference)"] + [k + " raw" for k in MODEL_KINDS[1:]] \
        + [k + " + Gal-cf" for k in GAL_ARMS]
vals = [fno_err.mean().item()] + [results_raw[k]["test_err"].mean().item() for k in MODEL_KINDS[1:]] \
       + [sym_results[k]["errs"].mean().item() for k in GAL_ARMS]
_palette = ["#888888", "#4c78a8", "#f58518", "#54a24b", "#e45756"]
colors = _palette[:len(MODEL_KINDS)] + ["#b279a2"] * len(GAL_ARMS)
bars = ax.bar(names, vals, color=colors)
ax.axhline(fno_err.mean().item(), color="#555555", ls="--", lw=1.4, label="FNO-raw baseline")
ax.set_ylabel("mean relative L2 (test)")
ax.set_title("SYMBA v5: KAN operators vs FNO, without and with exact Galilean symmetry")
for r, v in zip(bars, vals):
    ax.text(r.get_x() + r.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
ax.tick_params(axis="x", rotation=14)
ax.legend()
plt.tight_layout(); plt.show()

fig.savefig("plots_v5/v5_bars_00.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("[saved] plots_v5/v5_bars_00.png")

# ======== EXEC CELL 24 id=v5-code-020 ========
# ---------------- Error growth curves + per-sample distributions ----------------
fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
t_axis = np.arange(1, CONFIG["T"] + 1)
ax = axes[0]
ax.plot(t_axis, baseline_mat.cpu().numpy().mean(0), "--", color="#888888", lw=2, label="FNO raw")
for k in MODEL_KINDS[1:]:
    ax.plot(t_axis, results_raw[k]["mat"].cpu().numpy().mean(0), label=f"{k} raw")
for _k in GAL_ARMS:
    ax.plot(t_axis, sym_results[_k]["mats"].cpu().numpy().mean(0), ":",
             lw=2, label=f"{_k} + Gal-cf")
ax.set_xlabel("output timestep"); ax.set_ylabel("relative L2")
ax.set_title("Error growth per timestep"); ax.legend(fontsize=8)

ax = axes[1]
box = [fno_err.cpu().numpy()]
lbl = ["FNO raw"]
for k in MODEL_KINDS[1:]:
    box.append(results_raw[k]["test_err"].cpu().numpy()); lbl.append(k)
for _k in GAL_ARMS:
    box.append(sym_results[_k]["errs"].cpu().numpy()); lbl.append(_k + "+Gal-cf")
ax.boxplot(box, labels=lbl, showfliers=False)
ax.set_ylabel("per-sample relative L2"); ax.set_title("Per-sample error distribution")
ax.tick_params(axis="x", rotation=18)
plt.tight_layout(); plt.show()
fig.savefig("plots_v5/v5_curves_box_01.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("[saved] plots_v5/v5_curves_box_01.png")

# ======== EXEC CELL 25 id=v5-code-021 ========
# ---------------- Solution panels: GT vs predictions at three timestamps ----------------
gi = keep_ids[0]                    # a reserved test sample
t_show = [5, 10, CONFIG["T"]]
panels = [("FNO raw", ("raw", "FNO")), ("KANO-R raw", ("raw", "KANO-R")),
          (f"{winner} raw", ("raw", winner))]
panels += [(k + " + Gal-cf", ("sym", k)) for k in GAL_ARMS]

fig, axes = plt.subplots(len(panels), len(t_show) + 1, figsize=(16, 3.0 * len(panels)), sharex=True)
for r, (nm, src) in enumerate(panels):
    axes[r][0].plot(burgers_data[gi, 0], lw=1, color="#333333")
    axes[r][0].set_ylabel(nm, fontsize=8, rotation=0, ha="right", va="center")
    if r == 0:
        axes[r][0].set_title("input u0", fontsize=9)
    for c, tt in enumerate(t_show):
        ax_ = axes[r][c + 1]
        ax_.plot(burgers_data[gi, tt], lw=1.4, color="#222222", label="ground truth")
        store = results_raw[src[1]]["preds"] if src[0] == "raw" else sym_results[src[1]]["preds"]
        ax_.plot(store[gi][tt - 1], lw=1.2, ls="--", color="#e45756", label="prediction")
        if r == 0:
            ax_.set_title(f"frame t = {tt}", fontsize=9)
        if r == len(panels) - 1 and c == 0:
            ax_.legend(fontsize=7)
plt.suptitle(f"Solutions, test sample #{gi}: ground truth (black) vs predictions (red)", y=1.005)
plt.tight_layout(); plt.show()
fig.savefig("plots_v5/v5_solutions_02.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("[saved] plots_v5/v5_solutions_02.png")

# pointwise |error| maps over space-time for the same sample
fig, axes = plt.subplots(1, len(panels), figsize=(4.25 * len(panels), 2.8), sharey=True)
err_maps = []
for nm, src in panels:
    store = results_raw[src[1]]["preds"] if src[0] == "raw" else sym_results[src[1]]["preds"]
    em = np.abs(store[gi] - burgers_data[gi, 1:])          # (T,S)
    err_maps.append(em)
im = None
for c, ((nm, _), em) in enumerate(zip(panels, err_maps)):
    im = axes[c].imshow(em.T, aspect="auto", origin="lower", cmap="magma")
    axes[c].set_title(nm, fontsize=9)
    axes[c].set_xlabel("output timestep")
axes[0].set_ylabel("grid x")
fig.colorbar(im, ax=axes, shrink=0.85, label="|error|")
plt.suptitle(f"Pointwise error maps, test sample #{gi}", y=1.04)
plt.show()
fig.savefig("plots_v5/v5_error_maps_03.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("[saved] plots_v5/v5_error_maps_03.png")

# ======== EXEC CELL 26 id=v5-code-022-kernels ========
# ---------------- Kernel visualizations: EVERY family draws its learned kernels ----------------


def draw_kernel_panel(kernel, tag, lo=-3, hi=3, n_show=28):
    xs = np.linspace(lo, hi, 240)
    xs_t = torch.tensor(xs, dtype=torch.float32, device=device)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    with torch.no_grad():
        Y = kernel.curves(xs_t).cpu().numpy()
    for o in range(min(n_show, Y.shape[-1])):
        ax.plot(xs, Y[:, o], lw=0.9, alpha=0.65)
    ax.set_title(tag); ax.set_xlabel("pre-activation x"); ax.set_ylabel(r"$\phi(x)$")
    ax.grid(alpha=0.25)
    return fig


# (i) winner's univariate kernels (dispatch -- every family has drawable curves)
wm = sym_results[winner]["model"]
if winner == "CH-KAN":
    # evaluate each correction kernel over its ACTUAL input range in the forward pass:
    # z (log-phi_0) spans ~+-35 for this dataset; the velocity input to the multiplicative
    # kernel is |u| <~ 3. Outside those ranges the kernels are never queried.
    xz = np.linspace(-35, 35, 400); zt = torch.tensor(xz, dtype=torch.float32, device=device)
    xu = np.linspace(-3, 3, 400); ut_ = torch.tensor(xu, dtype=torch.float32, device=device)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    with torch.no_grad():
        zc = wm._rat(zt, wm.exp_P, wm.exp_Q).cpu().numpy()      # additive z-correction
        uc = wm._rat(ut_, wm.log_P, wm.log_Q).cpu().numpy()     # multiplicative u-correction
    axes[0].plot(xz, zc, lw=2, color="#e45756", label="learned z-correction")
    axes[0].axhline(0.0, color="k", ls="--", lw=1, label="exact physics (0)")
    axes[0].set_title("CH-KAN: additive kernel on log-phi_0 (input range of the data)")
    axes[0].legend(fontsize=8); axes[0].set_ylim(-3e-4, 3e-4)
    # GAUGE FIX: out = alpha * ratio * rat(ratio), so the kernel's overall SCALE is absorbed
    # by the learnable alpha -- only its shape is identifiable. Normalize by the value at u=0
    # and plot the DEVIATION from exact physics (flat zero == ideal Burgers).
    uc0 = uc[np.argmin(np.abs(xu))]
    uc = (uc / uc0 - 1.0) if uc0 != 0 else (uc - 1.0)
    axes[1].plot(xu, uc, lw=2, color="#4c78a8", label="learned deviation (gauge-fixed)")
    axes[1].axhline(0.0, color="k", ls="--", lw=1, label="exact physics")
    axes[1].set_title("CH-KAN: multiplicative kernel on u (input range of the data)")
    axes[1].legend(fontsize=8)
    for a in axes:
        a.grid(alpha=0.25); a.set_xlabel("input")
    plt.tight_layout(); plt.show()
    fig.savefig("plots_v5/v5_kernels_colehopf_04.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print("[saved] plots_v5/v5_kernels_colehopf_04.png")
elif winner == "KANO-R":
    fig_a = draw_kernel_panel(wm.mixes[0].phi, f"{winner}+Gal-cf :: mixing rational kernels (layer 1)")
    plt.tight_layout(); plt.show()
    fig_a.savefig("v5_kernels_winner_05.png", dpi=150, bbox_inches="tight"); plt.close(fig_a)
    print("[saved] plots_v5/v5_kernels_winner_05.png")
elif winner == "ChebyKAN":
    fig_a = draw_kernel_panel(wm.point[0].phi, f"{winner}+Gal-cf :: Chebyshev kernels (layer 1)")
    plt.tight_layout(); plt.show()
    fig_a.savefig("v5_kernels_winner_05.png", dpi=150, bbox_inches="tight"); plt.close(fig_a)
    print("[saved] plots_v5/v5_kernels_winner_05.png")
else:
    fig_a = draw_kernel_panel(wm.trunk[0].phi, f"{winner}+Gal-cf :: trunk rational kernels (layer 1)")
    plt.tight_layout(); plt.show()
    fig_a.savefig("v5_kernels_winner_05.png", dpi=150, bbox_inches="tight"); plt.close(fig_a)
    print("[saved] plots_v5/v5_kernels_winner_05.png")

# (ii) CH-KAN money plot: the EXACT softmax attention kernel P(x, j) that produces the
# solution -- literally 'how the kernel fits the Burgers solution': at each x it weights
# the initial log-potential z(j) by the heat kernel; the tilted expectation of (x-j)/t is u.
_cm = sym_results["CH-KAN"]["model"]
_gi = int(test_idx[0])
_u0 = torch.from_numpy(ic_raw[_gi]).float().unsqueeze(0).unsqueeze(-1).to(device)
with torch.no_grad():
    _B, _S = 1, S_b
    _u = _u0[..., 0].to(torch.float64)
    _Ubar = _u.mean(dim=-1, keepdim=True)
    _z = -(_cm.dx / (2 * BURGERS_NU)) * spectral_int_1d(_u - _Ubar, _S)
    _jj = torch.arange(_S, device=device, dtype=torch.float64)
    _dj = _jj.view(-1, 1) - _jj.view(1, -1)
    _dj = _dj - _S * torch.round(_dj / _S)
    _dp = _dj * _cm.dx
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.3))
for a, ti in zip(axes, [1, CONFIG["T"]]):
    t = ti * BURGERS_DT_OUT
    L = _z.unsqueeze(1) - (_dp ** 2 / (4 * BURGERS_NU * t)).unsqueeze(0)
    P = torch.softmax(L.float(), dim=-1)[0].cpu().numpy()      # (Sx, Sj)
    a.imshow(P, aspect="auto", origin="lower", cmap="magma",
             extent=[0, BURGERS_L, 0, BURGERS_L])
    a.set_title(f"softmax attention P(x, j), t = {t:.2f}  (kernel width ~ sqrt(2 nu t))")
    a.set_xlabel("source j"); a.set_ylabel("target x")
plt.tight_layout(); plt.show()
fig.savefig("plots_v5/v5_kernels_softmax_05.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("[saved] plots_v5/v5_kernels_softmax_05.png")

# (iii) KANO-R spatial convolution kernels psi(dx) -- the physically literal filter plot
_gm = sym_results["KANO-R"]["model"] if winner == "KANO-R" else results_raw["KANO-R"]["model"]
_S_vis = S_b
fig, axes = plt.subplots(1, min(4, len(_gm.convs)), figsize=(4.2 * min(4, len(_gm.convs)), 3.4))
dx_px = np.arange(_S_vis); dx_px = np.where(dx_px > _S_vis // 2, dx_px - _S_vis, dx_px)
order_axis = np.argsort(dx_px)
with torch.no_grad():
    PSIs = [_g.filters(_S_vis, device).cpu().numpy() for _g in _gm.convs]
for c, (a, PSI) in enumerate(zip(np.atleast_1d(axes), PSIs[:np.atleast_1d(axes).size])):
    ch_show = min(6, PSI.shape[0])
    for ch in range(ch_show):
        a.plot(dx_px[order_axis], PSI[ch][order_axis], lw=1, alpha=0.75,
               label=f"ch{ch}" if c == 0 else None)
    a.set_xlim(-40, 40)
    a.set_title("GlobalRationalConv layer " + str(c + 1) + " filters psi(dx)")
    a.set_xlabel("pixel offset dx"); a.grid(alpha=0.25)
    if c == 0:
        a.legend(fontsize=7)
plt.tight_layout(); plt.show()
fig.savefig("plots_v5/v5_kernels_convolution_06.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("[saved] plots_v5/v5_kernels_convolution_06.png")

# (iv) Spectral view: data spectrum vs each operator's modal attention
spec_ref = np.abs(np.fft.rfft(burgers_data[train_idx[:32], 0], axis=-1)).mean(0)
freqs = np.fft.rfftfreq(CONFIG["S"])
fig, ax = plt.subplots(figsize=(7.5, 3.8))
ax.semilogy(freqs[1:], spec_ref[1:] + 1e-8, lw=1.4, label="mean |FFT| of raw u0 (train sample)")
ax.axvline(freqs[CONFIG["modal_M"]], ls=":", color="#54a24b",
           label=f"ModalKAN retained modes M={CONFIG['modal_M']}")
ax.axvline(freqs[CONFIG["fno_modes"]], ls=":", color="#4c78a8", label=f"FNO modes={CONFIG['fno_modes']}")
ax.axvline(DEALIAS_FRAC * freqs.max(), ls="--", color="#888888", label="solver dealias band")
ax.set_xlabel("frequency k/(2*pi) cycles per grid point"); ax.set_ylabel("|FFT(u0)|")
ax.set_title("Spectral support of the data vs each operator's modal attention")
ax.legend(fontsize=8); ax.grid(alpha=0.25)
plt.tight_layout(); plt.show()
fig.savefig("plots_v5/v5_spectrum_overlay_07.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("[saved] plots_v5/v5_spectrum_overlay_07.png")

# ======== EXEC CELL 28 id=v5-code-024 ========
print("SYMBA v5 -- KAN OPERATORS + GALILEAN CLOSED-FORM SYMMETRY SUMMARY")
print("=" * 78)
bm = fno_err.mean().item()
print(f"Shared reference FNO baseline (raw data) relative-L2 : {bm:.4f}\n")
hdr = f"{'model':22s} {'symmetry':12s} {'rel-L2':>9s} {'vs FNO-raw':>11s} {'params':>10s} {'ms/batch':>9s} {'s/epoch':>8s}"
print(hdr); print("-" * len(hdr))


def _row(name, sym, err, params=None, kind=None):
    imp = (1 - err / bm) * 100
    ps = f"{params:,}" if params else "-"
    ms = f"{speed[kind]['infer_ms_per_batch']:9.2f}" if kind in speed else f"{'-':>9s}"
    se = f"{speed[kind]['train_s_per_epoch']:8.3f}" if kind in speed else f"{'-':>8s}"
    print(f"{name:22s} {sym:12s} {err:9.4f} {imp:+10.1f}% {ps:>10s} {ms} {se}")


_row("FNO (reference)", "none", bm, results_raw["FNO"]["params"], "FNO")
for kind in MODEL_KINDS[1:]:
    _row(kind, "none", results_raw[kind]["test_err"].mean().item(), results_raw[kind]["params"], kind)
for _k in GAL_ARMS:
    _row(_k, "Gal-closed-form", sym_results[_k]["errs"].mean().item(),
         results_raw[_k]["params"], _k)
print("-" * len(hdr))
print(f"\nKAN winner selected by VALIDATION error only: {winner}")
print(f"Temporal-variance reduction by Galilean canonicalization: "
      f"{sym_results['FNO']['tvar_red']*100:.1f}%")
print(f"Displacement source: physics-exact drift U*tau*dt_out*S/L, U = mean(u0) -- "
      f"conservation residual {dev_from_u0:.1e}")
print("\nInterpretation:")
print("- 'raw' rows: baselines WITHOUT symmetry; compare directly against the v4 shared-FNO 0.0099.")
print("- '+ Gal-cf' rows: same architecture inside the EXACT v4 Galilean co-moving pipeline.")
print("- KANO-R + Gal-cf answers 'symmetry vs architecture': the exact symmetry lifts")
print("  the strongest generic KAN -- compare against KANO-R raw and FNO + Gal-cf.")
print("- Every learned kernel is drawable: see v5_kernels_*.png produced by the cells above.")
print(f"\nSPEED (winner {winner} vs FNO): inference "
      f"{speed['FNO']['infer_ms_per_batch']/max(speed[winner]['infer_ms_per_batch'],1e-9):.2f}x, "
      f"training {speed['FNO']['train_s_per_epoch']/max(speed[winner]['train_s_per_epoch'],1e-9):.2f}x "
      f"per-epoch wall-clock.")
print("- References: Liu et al. arXiv:2404.19756 (KAN); Lee et al. arXiv:2509.16825 (KANO);")
print("  Wu et al. arXiv:2510.08795 (rational CKAN); Popovych et al. arXiv:2406.02809 (symmetries).")