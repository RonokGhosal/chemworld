"""
SPLIT-LATENT encoder (commander's spec): separate CONTROLLABLE causal state from
action-induced NOISE state.

  z_c -- controllable state. Its dynamics are forced to be LINEAR + BILINEAR in (z_c, control
         actions) -- the true chain is exactly that (gate<-a0 linear, m1<-gate*a1 bilinear,
         m2<-m1, m3<-m2 linear), so this structured forward pushes z_c toward a LINEARLY
         recoverable latent. The noise innovation cannot fit linear-bilinear dynamics, so it
         is pushed out of z_c.
  z_n -- noise / variance state.

Crucial rule: the noise knob aN is itself an ACTION, so inverse dynamics must NOT see it (or
it would drag the noise response into z_c). Inverse predicts only the CONTROL actions (a0,a1)
from z_c. The variance head may use z_n + all actions (incl. aN). We also penalize z_c<->z_n
dependence and reward z_n (not z_c) for explaining the reconstruction residual.

Acceptance: linear R2(m1,m2,m3) from z_c >= 0.80; low R2(noise) from z_c; z_n explains noise.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .messy_world import MessyWorld, M1, M2, M3, N, A0, A1, AN
from .repr_encoder import collect, r2_multi


def _mlp(sizes):
    L = []
    for i in range(len(sizes) - 1):
        L.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            L.append(nn.SiLU())
    return nn.Sequential(*L)


CTRL = [A0, A1]            # control actions (inverse may use); aN is the noise knob (excluded)


class SplitModel(nn.Module):
    def __init__(self, obs_dim, dc=4, dn=2, n_act=3, h=128, bilinear=True, inv_all=False):
        super().__init__()
        self.dc, self.dn, self.nc = dc, dn, len(CTRL)
        self.bilinear, self.inv_all = bilinear, inv_all
        self.enc = _mlp([obs_dim, h, h, dc + dn])
        self.A = nn.Linear(dc, dc, bias=False)               # z_c linear dynamics
        self.B = nn.Linear(self.nc, dc, bias=False)          # control action drive
        self.Bil = nn.Linear(dc * self.nc, dc, bias=False) if bilinear else None  # bilinear gate
        self.var = _mlp([dn + n_act, h, dc])                 # log-var per z_c dim (z_n + actions)
        self.dec = _mlp([dc + dn, h, h, obs_dim])            # reconstruct obs from [z_c, z_n]
        self.inv = _mlp([2 * dc, h, n_act if inv_all else self.nc])  # actions from z_c

    def encode(self, o):
        z = self.enc(o)
        return z[..., :self.dc], z[..., self.dc:]

    def forward_c(self, zc, a_ctrl):
        out = self.A(zc) + self.B(a_ctrl)
        if self.bilinear:                                    # the structured AND-gate term
            bil = (zc.unsqueeze(-1) * a_ctrl.unsqueeze(-2)).flatten(-2)
            out = out + self.Bil(bil)
        return out


def _vic(z):
    std = torch.sqrt(z.var(0) + 1e-4)
    v = torch.relu(1.0 - std).mean()
    zc = z - z.mean(0)
    cov = (zc.T @ zc) / (len(z) - 1)
    off = cov - torch.diag(torch.diag(cov))
    return v, (off ** 2).sum() / z.shape[1]


def _cross_cov(zc, zn):
    zc = zc - zc.mean(0); zn = zn - zn.mean(0)
    c = (zc.T @ zn) / (len(zc) - 1)
    return (c ** 2).sum() / zc.shape[1]


def train_split(Ob, A, Oa, dc=4, dn=2, epochs=8000, lr=1e-3, seed=0,
                recon_w=1.0, inv_w=1.0, vic_w=2.0, cross_w=2.0,
                bilinear=True, inv_all=False, use_zn=True):
    """Ablation flags (all default to the full model):
       bilinear=False -> linear-only forward (drop the structured AND-gate term)
       inv_all=True   -> inverse predicts ALL actions incl. the noise knob aN (not control-only)
       use_zn=False   -> no noise latent (dn=0)
       cross_w=0      -> no z_c/z_n decorrelation penalty"""
    torch.manual_seed(seed)
    if not use_zn:
        dn = 0
    Ot = torch.tensor(Ob); On = torch.tensor(Oa); At = torch.tensor(A)
    Act = At[:, CTRL]
    inv_tgt = At if inv_all else Act
    m = SplitModel(Ob.shape[1], dc, dn, bilinear=bilinear, inv_all=inv_all)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for ep in range(epochs):
        zc, zn = m.encode(Ot); zcn, znn = m.encode(On)
        mu = m.forward_c(zc, Act)
        logv = m.var(torch.cat([zn, At], -1)).clamp(-6, 6)
        fwd = 0.5 * ((zcn - mu) ** 2 / torch.exp(logv) + logv)
        fwd = (torch.exp(logv).detach() ** 0.5 * fwd).mean()          # beta-NLL
        recon = ((m.dec(torch.cat([zc, zn], -1)) - Ot) ** 2).mean()   # z_n captures residual
        inv = ((inv_tgt - m.inv(torch.cat([zc, zcn], -1))) ** 2).mean()
        vc, cc = _vic(zc)
        if dn > 0:
            vn, cn = _vic(zn); cross = _cross_cov(zc, zn)
        else:
            vn = cn = cross = torch.zeros((), dtype=zc.dtype)
        loss = (fwd + recon_w * recon + inv_w * inv + vic_w * (vc + vn) + cc + cn
                + cross_w * cross)
        opt.zero_grad(); loss.backward(); opt.step()
    return m


def main(seed=0, n=4000, dc=4, dn=2):
    rng = np.random.default_rng(seed)
    w = MessyWorld(rng, obs_dim=14, nonlinear=True); w.reset()
    Ob, A, Oa, Zb, Za = collect(w, n, rng)
    tgt = {"m1": Za[:, M1], "m2": Za[:, M2], "m3": Za[:, M3], "noise_n": Za[:, N]}
    m = train_split(Ob, A, Oa, dc=dc, dn=dn, seed=seed)
    with torch.no_grad():
        zc, zn = m.encode(torch.tensor(Oa))
    zc, zn = zc.numpy(), zn.numpy()
    print("=" * 78)
    print("SPLIT LATENT -- z_c (controllable) vs z_n (noise)   [linear R2; bar: chain>=0.80]")
    print("=" * 78)
    rc = r2_multi(zc, tgt); rn = r2_multi(zn, tgt)
    chain = np.mean([rc["m1"], rc["m2"], rc["m3"]])
    print(f"  z_c recovers chain:  m1={rc['m1']:.2f} m2={rc['m2']:.2f} m3={rc['m3']:.2f}  "
          f"avg={chain:.2f}   (BAR >= 0.80)")
    print(f"  z_c recovers noise:  R2(noise)={rc['noise_n']:.2f}   (want LOW)")
    print(f"  z_n recovers noise:  R2(noise)={rn['noise_n']:.2f}   (want HIGH)")
    print(f"  z_n recovers chain:  avg={np.mean([rn['m1'],rn['m2'],rn['m3']]):.2f}   (want low)")
    ok = chain >= 0.80 and rc["noise_n"] < 0.3 and rn["noise_n"] > 0.5
    print(f"\n  ACCEPTANCE (chain>=0.80, z_c-noise<0.3, z_n-noise>0.5): {'PASS' if ok else 'NOT YET'}")
    print("=" * 78)
    return chain, rc["noise_n"], rn["noise_n"]


if __name__ == "__main__":
    main()
