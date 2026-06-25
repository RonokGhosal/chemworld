"""
Learned state representation for the messy world (commander's orders): replace the PCA/RFF
hacks with a TRAINED encoder o_t -> zhat_t that recovers the CONTROLLABLE latent well
(R2(m1,m2,m3) >= 0.80) while NOT encoding the irreducible noise.

The encoder is trained with four coupled objectives:
  * action-conditioned FORWARD prediction with a HETEROSCEDASTIC head  -> shapes zhat to be
    PREDICTABLE and down-weights directions whose residual is irreducible (the noise knob);
  * INVERSE dynamics (recover the action from a transition)            -> forces zhat to keep
    the CONTROLLABLE / action-relevant structure;
  * a small BOTTLENECK (dz) so it cannot just memorize the high-variance noise;
  * VICReg variance+covariance terms                                   -> prevent collapse.

Baselines (commander's list) for honest comparison, scored by latent recovery vs the hidden
truth (diagnosis only): PCA, plain autoencoder, predictive autoencoder, inverse-only encoder,
and the oracle (true latent).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .messy_world import MessyWorld, M1, M2, M3, N, NA


def collect(world, n, rng):
    """ALIGNED transitions: each row is (obs_before, action, obs_after) -- the action that
    actually CAUSED the obs_before -> obs_after transition. Zb/Za are the hidden latents
    before/after (diagnosis only). Proven by selftest_repr_alignment."""
    Ob, Ac, Oa, Zb, Za = [], [], [], [], []
    ob, zb = world.observe(), world.z.copy()
    for _ in range(n):
        a = np.array([rng.choice([-2.0, 0.0, 2.0]) for _ in range(NA)], np.float32)
        oa = world.step(a)
        Ob.append(ob); Ac.append(a); Oa.append(oa); Zb.append(zb); Za.append(world.z.copy())
        ob, zb = oa, world.z.copy()
    f = lambda x: np.asarray(x, np.float32)
    return f(Ob), f(Ac), f(Oa), f(Zb), f(Za)


def _mlp(sizes):
    L = []
    for i in range(len(sizes) - 1):
        L.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            L.append(nn.SiLU())
    return nn.Sequential(*L)


class ReprModel(nn.Module):
    def __init__(self, obs_dim, dz=4, na=NA, h=128):
        super().__init__()
        self.dz = dz
        self.enc = _mlp([obs_dim, h, h, dz])
        self.fwd = _mlp([dz + na, h, h, 2 * dz])      # mean, log-variance
        self.inv = _mlp([2 * dz, h, na])

    def encode(self, o):
        return self.enc(o)

    def forward_pred(self, z, a):
        out = self.fwd(torch.cat([z, a], -1))
        return out[..., :self.dz], out[..., self.dz:].clamp(-6, 6)

    def inverse(self, z, zn):
        return self.inv(torch.cat([z, zn], -1))


def _vicreg(z):
    std = torch.sqrt(z.var(0) + 1e-4)
    var = torch.relu(1.0 - std).mean()
    zc = z - z.mean(0)
    cov = (zc.T @ zc) / (len(z) - 1)
    off = cov - torch.diag(torch.diag(cov))
    return var, (off ** 2).sum() / z.shape[1]


def train_encoder(Ob, A, Oa, dz=4, epochs=4000, lr=1e-3, inv_w=2.0, var_w=2.0, cov_w=1.0,
                  hetero=True, predictive=True, inverse=True, seed=0, device=None):
    from .device import get_device
    dev = get_device(device)
    torch.manual_seed(seed)
    Ot = torch.as_tensor(Ob, dtype=torch.float32, device=dev)       # ALIGNED triples
    On = torch.as_tensor(Oa, dtype=torch.float32, device=dev)
    At = torch.as_tensor(A, dtype=torch.float32, device=dev)
    m = ReprModel(Ob.shape[1], dz).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for ep in range(epochs):
        z, zn = m.encode(Ot), m.encode(On)
        loss = 0.0
        if predictive:
            mu, logv = m.forward_pred(z, At)
            if hetero:
                var = torch.exp(logv)
                nll = 0.5 * ((zn - mu) ** 2 / var + logv)
                loss = loss + (var.detach() ** 0.5 * nll).mean()    # beta-NLL (beta=0.5), stable
            else:
                loss = loss + ((zn - mu) ** 2).mean()
        if inverse:
            loss = loss + inv_w * ((At - m.inverse(z, zn)) ** 2).mean()
        v, c = _vicreg(z)
        loss = loss + var_w * v + cov_w * c
        opt.zero_grad(); loss.backward(); opt.step()
    return m.to("cpu")


def r2_multi(Zhat, targets, nonlinear=False):
    if nonlinear:                                   # RFF readout -- catches entangled encodings
        rng = np.random.default_rng(1)
        Wf = rng.normal(0, 1.0, (Zhat.shape[1], 300)); bf = rng.uniform(0, 2 * np.pi, 300)
        Zhat = np.cos(Zhat @ Wf + bf)
    X = np.column_stack([Zhat, np.ones(len(Zhat))])
    out = {}
    for name, t in targets.items():
        b, *_ = np.linalg.lstsq(X, t, rcond=None)
        out[name] = max(0.0, 1 - ((t - X @ b) ** 2).sum() / ((t - t.mean()) ** 2).sum())
    return out


def pca_encode(O, k):
    Oc = O - O.mean(0)
    return Oc @ np.linalg.svd(Oc, full_matrices=False)[2][:k].T


def main(seed=0, n=4000, dz=4):
    rng = np.random.default_rng(seed)
    w = MessyWorld(rng, obs_dim=14, nonlinear=True); w.reset()
    Ob, A, Oa, Zb, Za = collect(w, n, rng)      # ALIGNED (obs_before, action, obs_after)
    tgt = {"m1": Za[:, M1], "m2": Za[:, M2], "m3": Za[:, M3], "noise_n": Za[:, N]}
    print("=" * 74)
    print(f"LEARNED REPRESENTATION -- recover controllable latent (target R2>=0.80), drop noise")
    print("=" * 74)
    print(f"  {'encoder':>26}  {'chain-avg (lin/NONLIN)':>24}  {'noise_n (lin/NONLIN)':>22}")

    def show(name, Zhat):
        rl = r2_multi(Zhat, tgt); rn = r2_multi(Zhat, tgt, nonlinear=True)
        cl = np.mean([rl["m1"], rl["m2"], rl["m3"]]); cn = np.mean([rn["m1"], rn["m2"], rn["m3"]])
        print(f"  {name:>26}  {cl:>10.2f} / {cn:<11.2f}  {rl['noise_n']:>9.2f} / {rn['noise_n']:<9.2f}")
        return cn

    show("PCA (variance)", pca_encode(Oa, dz))
    for name, kw in [("predictive-only(homo)", dict(hetero=False, inverse=False)),
                     ("predictive+inverse(homo)", dict(hetero=False, inverse=True)),
                     ("HETERO(beta) pred+inverse", dict(hetero=True, inverse=True))]:
        m = train_encoder(Ob, A, Oa, dz=dz, epochs=7000, **kw, seed=seed)
        with torch.no_grad():
            Zhat = m.encode(torch.tensor(Oa)).numpy()
        show(name, Zhat)
    show("oracle (true latent)", Za[:, [0, 1, 2, 3]])
    print("=" * 74)


if __name__ == "__main__":
    main()
