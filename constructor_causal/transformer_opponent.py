"""
The TRANSFORMER OPPONENT -- a GPT-style next-STATE predictor (our world's "next-token model").

This is the prediction-trained net we aim to BEAT on interventional inference, and it is the
cleanest, most honest GPU load in the project: scale d_model / layers and it saturates a GPU.
Trained on PASSIVE trajectories (random actions) -> it learns CORRELATIONS, including the
confounder's spurious one. Later we ask it interventional / counterfactual queries and compare
to the active-intervention causal agent.

Decoder-only, causal mask: input token t = [state_t, action_t]; target = state_{t+1}.
"""
from __future__ import annotations

import contextlib

import numpy as np
import torch
import torch.nn as nn

from .device import get_device


class Block(nn.Module):
    def __init__(self, d, h, p=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True, dropout=p)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mask):
        y = self.ln1(x)
        a, _ = self.attn(y, y, y, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class StateTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, d=256, h=8, layers=6, max_len=128):
        super().__init__()
        self.inp = nn.Linear(state_dim + act_dim, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, state_dim)

    def forward(self, states, actions):
        T = states.shape[1]
        x = self.inp(torch.cat([states, actions], -1))
        x = x + self.pos(torch.arange(T, device=states.device))[None]
        # boolean causal mask (True = blocked). Avoids the MPS -inf-mask NaN bug.
        mask = torch.ones(T, T, dtype=torch.bool, device=states.device).triu(1)
        for b in self.blocks:
            x = b(x, mask)
        return self.head(self.ln(x))                 # (B, T, state_dim): predicted NEXT state


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def train(model, S, A, epochs=200, lr=3e-4, batch=64, device=None, log_every=50, amp=True):
    """amp=True enables fp16 mixed precision ON CUDA (uses tensor cores -> the real GPU win).
    No-op on cpu/mps, so the same call is correct everywhere."""
    dev = get_device(device)
    model = model.to(dev)
    use_amp = amp and dev.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    S = torch.as_tensor(S, dtype=torch.float32)
    A = torch.as_tensor(A, dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    N = len(S)
    for ep in range(epochs):
        idx = torch.randint(0, N, (min(batch, N),))
        s = S[idx].to(dev); a = A[idx].to(dev)
        opt.zero_grad()
        ctx = torch.autocast("cuda", dtype=torch.float16) if use_amp else contextlib.nullcontext()
        with ctx:
            pred = model(s, a)                        # predict next state at each position
            loss = ((pred[:, :-1] - s[:, 1:]) ** 2).mean()
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        if log_every and (ep % log_every == 0 or ep == epochs - 1):
            print(f"    ep {ep:>4}  loss {loss.item():.4f}  [{dev}{' amp' if use_amp else ''}]")
    return model.to("cpu")


if __name__ == "__main__":
    import sys, time
    from .bigworld import BigWorld, collect_trajectories

    n_vars = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    d = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    layers = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    batch = int(sys.argv[4]) if len(sys.argv) > 4 else 256        # bigger batch saturates a GPU
    rng = np.random.default_rng(0)
    w = BigWorld(n_vars=n_vars, n_act=8, rng=rng)
    S, A = collect_trajectories(w, n_traj=512, T=64, rng=rng)
    print(f"world n_vars={n_vars}  data S{tuple(S.shape)} A{tuple(A.shape)}  batch={batch}")
    m = StateTransformer(n_vars, 8, d=d, h=8, layers=layers, max_len=64)
    print(f"StateTransformer d={d} layers={layers} -> {n_params(m):,} params  device={get_device()}")
    t = time.time()
    train(m, S, A, epochs=200, batch=batch, device=None)
    dt = time.time() - t
    print(f"trained 200 steps in {dt:.1f}s  ({1000*dt/200:.0f} ms/step)")
