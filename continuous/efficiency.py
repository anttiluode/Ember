"""
Continuous-signal efficiency test.

The economy claim, stated as a Pareto question: on a long, mostly-predictable stream
with rare persistent regime changes, can the surprise-gated cell hold prediction
accuracy while running the expensive antisymmetric flow on only a FRACTION of steps?

Four parameter-matched configs, all predicting the CLEAN next observation from a
NOISY stream of a switching oscillator:

    S-only        the Ember cell with the arrow OFF (g=0): symmetric recurrence only.
                  The floor. If the arrow is worthless, this already wins.
    Ember-uniform the arrow ON every step (g=1). Full cost. The accuracy ceiling the
                  arrow can buy.
    Ember-gated   the arrow gated by surprise. Should approach uniform's accuracy at a
                  fraction of uniform's arrow-compute.
    GRU           external reference.

WIN requires three legs: the arrow must actually help (uniform beats S-only), the gate
must keep that help (gated ~ uniform), and it must be cheap (mean gate well below 1).
Any leg that fails is reported as such — including "the arrow doesn't help here."
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import autograd.numpy as anp
from autograd import value_and_grad

from ember import cell, baselines, continuous
from ember.optim import Adam, count_params


def train(loss_fn, params, gen, steps, lr, label=""):
    opt = Adam(params, lr=lr)
    vg = value_and_grad(loss_fn)
    t0 = time.time()
    for i in range(steps):
        batch = next(gen)
        L, g = vg(params, *batch)
        params = opt.step(params, g)
        if (i + 1) % max(1, steps // 4) == 0:
            print(f"    [{label}] step {i+1:4d}/{steps}  loss={float(L):.5f}")
    print(f"    [{label}] trained in {time.time()-t0:.1f}s")
    return params


def clean_mse_ember(p, Xn, Xc, **kw):
    out = cell.forward(p, Xn, **kw)
    return float(np.mean((np.asarray(out["preds"])[:, :-1] - Xc[:, 1:]) ** 2))


def clean_mse_gru(p, Xn, Xc):
    out = baselines.forward(p, Xn)
    return float(np.mean((np.asarray(out["preds"])[:, :-1] - Xc[:, 1:]) ** 2))


def ember_loss(p, Xn, Xc, sw, gate_on=True, force=None, lam=0.0):
    out = cell.forward(p, Xn, gate_on=gate_on, force=force)
    mse = anp.mean((out["preds"][:, :-1] - Xc[:, 1:]) ** 2)
    return mse + lam * anp.mean(out["gate"])


def gru_loss(p, Xn, Xc, sw):
    out = baselines.forward(p, Xn)
    return anp.mean((out["preds"][:, :-1] - Xc[:, 1:]) ** 2)


def gate_vs_switch(p, Xn, sw, warmup=4):
    out = cell.forward(p, Xn, gate_on=True)
    gate = np.asarray(out["gate"])
    m = np.zeros_like(sw); m[:, warmup:] = True
    gj = gate[sw & m].mean() if (sw & m).any() else 0.0
    gn = gate[(~sw) & m].mean()
    return float(gate[m].mean()), float(gj / (gn + 1e-9))


def main():
    print("=" * 64)
    print("EXPERIMENT 03  —  continuous-signal efficiency (switching oscillator)")
    print("=" * 64)
    obs_dim, T = 4, 96
    Xn, Xc, sw = continuous.generate_switching(512, T, obs_dim, switch_prob=0.03,
                                               noise=0.30, seed=1)
    Xnt, Xct, swt = continuous.generate_switching(256, T, obs_dim, switch_prob=0.03,
                                                  noise=0.30, seed=2)
    naive = float(np.mean((Xnt[:, :-1] - Xct[:, 1:]) ** 2))  # predict noisy current as next
    print(f"\ndata: {Xn.shape[0]} train / {Xnt.shape[0]} test, T={T}, noise=0.30, "
          f"switch rate={sw.mean():.3f}")
    print(f"naive baseline (copy current) clean-MSE = {naive:.4f}")

    STEPS, BATCH, LR = 500, 24, 4e-3
    pe = cell.init_params(obs_dim=obs_dim, D=48, K=4, seed=0)
    pg = baselines.init_params(obs_dim=obs_dim, H=38, seed=0)
    print(f"params: Ember={count_params(pe)}  GRU={count_params(pg)}")

    rng = lambda s: np.random.default_rng(s)
    g_sonly = continuous.batches(Xn, Xc, sw, BATCH, rng(10))
    g_unif = continuous.batches(Xn, Xc, sw, BATCH, rng(11))
    g_gate = continuous.batches(Xn, Xc, sw, BATCH, rng(12))
    g_gru = continuous.batches(Xn, Xc, sw, BATCH, rng(13))

    print("\n-- S-only (arrow off) --")
    p_s = train(lambda p, a, b, c: ember_loss(p, a, b, c, force=0.0),
                {k: v.copy() for k, v in pe.items()}, g_sonly, STEPS, LR, "s-only")
    print("\n-- Ember-uniform (arrow every step) --")
    p_u = train(lambda p, a, b, c: ember_loss(p, a, b, c, gate_on=False),
                {k: v.copy() for k, v in pe.items()}, g_unif, STEPS, LR, "uniform")
    print("\n-- Ember-gated (arrow on surprise, priced) --")
    p_g = train(lambda p, a, b, c: ember_loss(p, a, b, c, gate_on=True, lam=0.02),
                {k: v.copy() for k, v in pe.items()}, g_gate, STEPS, LR, "gated")
    print("\n-- GRU --")
    p_r = train(gru_loss, pg, g_gru, STEPS, LR, "gru")

    mse_s = clean_mse_ember(p_s, Xnt, Xct, force=0.0)
    mse_u = clean_mse_ember(p_u, Xnt, Xct, gate_on=False)
    mse_g = clean_mse_ember(p_g, Xnt, Xct, gate_on=True)
    mse_r = clean_mse_gru(p_r, Xnt, Xct)
    mean_gate, conc = gate_vs_switch(p_g, Xnt, swt)

    print("\n" + "-" * 64)
    print("RESULTS  (clean next-step MSE, lower better; arrow-frac = mean gate)")
    print(f"  S-only  (arrow off)   MSE = {mse_s:.4f}   arrow-frac = 0.00")
    print(f"  Ember-uniform (g=1)   MSE = {mse_u:.4f}   arrow-frac = 1.00")
    print(f"  Ember-gated           MSE = {mse_g:.4f}   arrow-frac = {mean_gate:.2f}")
    print(f"  GRU (reference)       MSE = {mse_r:.4f}")
    print(f"\n  gate concentration on regime switches = {conc:.2f}x")

    arrow_helps = mse_u < mse_s * 0.97
    gated_keeps = mse_g <= mse_u * 1.05
    cheap = mean_gate < 0.5
    print("\n" + "-" * 64)
    print("LEDGER")
    print(f"  [{'V' if arrow_helps else 'K'}] the arrow helps (uniform beats S-only by >3%): "
          f"{arrow_helps}  (S-only {mse_s:.4f} -> uniform {mse_u:.4f})")
    print(f"  [{'V' if gated_keeps else 'K'}] gating keeps the accuracy (gated <= 1.05x uniform): "
          f"{gated_keeps}")
    print(f"  [{'V' if cheap else 'K'}] and it is cheap (arrow runs <50% of steps): "
          f"{cheap}  (arrow-frac {mean_gate:.2f})")
    verdict = "EFFICIENCY WIN" if (arrow_helps and gated_keeps and cheap) else "PARTIAL / KILL"
    print(f"\n  VERDICT: {verdict}")
    print("=" * 64)


if __name__ == "__main__":
    main()
