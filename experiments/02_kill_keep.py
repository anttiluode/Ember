"""
Experiment 02 — the kill/keep test (protected-memory regime).

Task: running PARITY of rare phase-resets in a rotating carrier. To solve it a
model must detect each reset, flip a protected bit, and hold it for free until the
next one. This is the regime the smooth next-step task could NOT test, because
there the cheap symmetric half already solved everything.

Three models, matched on parameter count:
    Ember          S(+)A excitable cell, gate ON, arrow priced (lam>0)
    Ember-uniform  SAME cell, gate forced to 1 (winding advances EVERY step)
    GRU            standard stateful baseline

Two honest questions:
  1. ACCURACY — does Ember solve parity as well as the GRU?
  2. MECHANISM — does the gate fire on the resets and stay dark between them,
                 and does removing the gate (uniform) BREAK the memory?

KEEP requires: Ember accurate AND gate concentrates on surprise AND the uniform
ablation is meaningfully worse (so the gate is load-bearing, not decoration).
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from autograd import value_and_grad

from ember import cell, baselines, data
from ember.optim import Adam, count_params, sigmoid as _sig


def sigmoid_np(x):
    return 0.5 * (np.tanh(0.5 * x) + 1.0)


def train(loss_fn, params, gen, steps, lr, label=""):
    opt = Adam(params, lr=lr)
    vg = value_and_grad(loss_fn)
    t0 = time.time()
    for i in range(steps):
        batch = next(gen)
        L, g = vg(params, *batch)
        params = opt.step(params, g)
        if (i + 1) % max(1, steps // 5) == 0:
            print(f"    [{label}] step {i+1:4d}/{steps}  loss={float(L):.5f}")
    print(f"    [{label}] trained in {time.time()-t0:.1f}s")
    return params


def parity_acc_ember(p, X, Y, gate_on, warmup=4):
    out = cell.forward(p, X, gate_on=gate_on)
    pred = (sigmoid_np(np.asarray(out["task"])) > 0.5).astype(float)
    m = np.zeros_like(Y, dtype=bool); m[:, warmup:] = True
    return float((pred[m] == Y[m]).mean())


def parity_acc_gru(p, X, Y, warmup=4):
    out = baselines.forward(p, X)
    pred = (sigmoid_np(np.asarray(out["task"])) > 0.5).astype(float)
    m = np.zeros_like(Y, dtype=bool); m[:, warmup:] = True
    return float((pred[m] == Y[m]).mean())


def economy(p, X, jumps, warmup=4):
    out = cell.forward(p, X, gate_on=True)
    gate = np.asarray(out["gate"])
    jb = jumps.copy()
    mask = np.zeros_like(jb); mask[:, warmup:] = True
    gj = gate[jb & mask].mean()
    gn = gate[(~jb) & mask].mean()
    return {"g_jump": float(gj), "g_non": float(gn),
            "g_ratio": float(gj / (gn + 1e-9)), "mean_gate": float(gate[mask].mean())}


def main():
    print("=" * 64)
    print("EXPERIMENT 02  —  kill/keep, parity-of-surprises (protected memory)")
    print("=" * 64)
    obs_dim, T = 4, 64
    Xtr, Ytr, Jtr = data.generate_parity(n_seqs=640, T=T, obs_dim=obs_dim, p_jump=0.06, seed=1)
    Xte, Yte, Jte = data.generate_parity(n_seqs=256, T=T, obs_dim=obs_dim, p_jump=0.06, seed=2)
    print(f"\ndata: {Xtr.shape[0]} train / {Xte.shape[0]} test, T={T}, "
          f"surprise rate={Jtr.mean():.3f}")

    STEPS, BATCH, LR = 900, 32, 4e-3
    rng = np.random.default_rng(0)
    pe = cell.init_params(obs_dim=obs_dim, D=48, K=4, seed=0)
    pg = baselines.init_params(obs_dim=obs_dim, H=38, seed=0)
    print(f"params: Ember={count_params(pe)}  GRU={count_params(pg)}")

    gen_e = data.batches_xy(Xtr, Ytr, Jtr, BATCH, rng)
    gen_u = data.batches_xy(Xtr, Ytr, Jtr, BATCH, np.random.default_rng(1))
    gen_g = data.batches_xy(Xtr, Ytr, Jtr, BATCH, np.random.default_rng(2))

    print("\n-- Ember (gate ON, arrow priced) --")
    pe_on = train(lambda p, X, Y, J: cell.parity_loss(p, X, Y, gate_on=True, lam=0.02),
                  {k: v.copy() for k, v in pe.items()}, gen_e, STEPS, LR, "ember")
    print("\n-- Ember-uniform (gate=1) --")
    pe_un = train(lambda p, X, Y, J: cell.parity_loss(p, X, Y, gate_on=False, lam=0.0),
                  {k: v.copy() for k, v in pe.items()}, gen_u, STEPS, LR, "uniform")
    print("\n-- GRU --")
    pg_t = train(lambda p, X, Y, J: baselines.parity_loss(p, X, Y),
                 pg, gen_g, STEPS, LR, "gru")

    acc_on = parity_acc_ember(pe_on, Xte, Yte, gate_on=True)
    acc_un = parity_acc_ember(pe_un, Xte, Yte, gate_on=False)
    acc_gru = parity_acc_gru(pg_t, Xte, Yte)
    eco = economy(pe_on, Xte, Jte)
    base_rate = max(Yte.mean(), 1 - Yte.mean())  # majority-class baseline

    print("\n" + "-" * 64)
    print(f"RESULTS  (parity accuracy; majority-class baseline = {base_rate:.3f})")
    print(f"  Ember (gate ON)      acc = {acc_on:.3f}")
    print(f"  Ember-uniform (g=1)  acc = {acc_un:.3f}")
    print(f"  GRU                  acc = {acc_gru:.3f}")
    print("\nECONOMY  (Ember gate vs ground-truth surprise)")
    print(f"  mean gate over all steps      = {eco['mean_gate']:.3f}")
    print(f"  gate on surprise / on quiet   = {eco['g_ratio']:.2f}x  "
          f"(jump={eco['g_jump']:.3f}, quiet={eco['g_non']:.3f})")

    accurate = acc_on >= 0.90
    concentrates = eco['g_ratio'] >= 2.0
    gate_matters = (acc_on - acc_un) >= 0.10
    print("\n" + "-" * 64)
    print("LEDGER")
    print(f"  [{'V' if accurate else 'K'}] Ember solves parity (acc>=0.90): {accurate}")
    print(f"  [{'V' if concentrates else 'K'}] gate concentrates on surprise (>=2x): {concentrates}")
    print(f"  [{'V' if gate_matters else 'K'}] gate is load-bearing "
          f"(gated - uniform >= 0.10): {gate_matters}  "
          f"(Δ={acc_on - acc_un:+.3f})")
    verdict = "KEEP" if (accurate and concentrates and gate_matters) else "PARTIAL / KILL"
    print(f"\n  VERDICT: {verdict}")
    print("=" * 64)


if __name__ == "__main__":
    main()
