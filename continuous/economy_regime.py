"""
Where does the economy pay? — a noise sweep.

experiment 03 (efficiency.py) found a clean boundary by accident: the surprise-gate
only saves compute when the expensive arrow is needed SPARSELY in time. On a noisy
stream the arrow's value is continuous (it denoises every step), so gating it to
surprises throws the benefit away.

This sweeps the observation noise and, at each level, asks the three efficiency legs:
does the arrow help, does gating keep the accuracy, and is it cheap. The prediction:
at LOW noise the arrow's value concentrates at the rare regime switches (gate can
sparsify it -> win); at HIGH noise it is a continuous burden (gate cannot -> no win).

Lighter settings than efficiency.py (fewer steps/seqs, no GRU) so the whole sweep
runs in one sitting. The map, not the decimal, is the point.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import autograd.numpy as anp
from autograd import value_and_grad

from ember import cell, continuous
from ember.optim import Adam


def train(loss_fn, params, gen, steps, lr):
    opt = Adam(params, lr=lr)
    vg = value_and_grad(loss_fn)
    for i in range(steps):
        b = next(gen)
        L, g = vg(params, *b)
        params = opt.step(params, g)
    return params


def ember_loss(p, Xn, Xc, sw, gate_on=True, force=None, lam=0.0):
    out = cell.forward(p, Xn, gate_on=gate_on, force=force)
    return anp.mean((out["preds"][:, :-1] - Xc[:, 1:]) ** 2) + lam * anp.mean(out["gate"])


def clean_mse(p, Xn, Xc, **kw):
    out = cell.forward(p, Xn, **kw)
    return float(np.mean((np.asarray(out["preds"])[:, :-1] - Xc[:, 1:]) ** 2))


def gate_stats(p, Xn, sw, warmup=4):
    g = np.asarray(cell.forward(p, Xn, gate_on=True)["gate"])
    m = np.zeros_like(sw); m[:, warmup:] = True
    gj = g[sw & m].mean() if (sw & m).any() else 0.0
    gn = g[(~sw) & m].mean()
    return float(g[m].mean()), float(gj / (gn + 1e-9))


def main():
    print("=" * 70)
    print("SWEEP  —  where does the surprise-economy pay? (observation noise)")
    print("=" * 70)
    obs_dim, T = 4, 56
    STEPS, BATCH, LR = 200, 24, 4e-3
    pe = cell.init_params(obs_dim=obs_dim, D=48, K=4, seed=0)

    print(f"\n{'noise':>6} | {'S-only':>7} {'uniform':>7} {'gated':>7} | "
          f"{'arrowfrac':>9} {'conc':>5} | {'helps':>5} {'keeps':>5} {'cheap':>5} | verdict")
    print("-" * 70)
    rows = []
    for noise in [0.0, 0.10, 0.30]:
        Xn, Xc, sw = continuous.generate_switching(224, T, obs_dim, switch_prob=0.035,
                                                   noise=noise, seed=1)
        Xnt, Xct, swt = continuous.generate_switching(128, T, obs_dim, switch_prob=0.035,
                                                      noise=noise, seed=2)
        t0 = time.time()
        p_s = train(lambda p, a, b, c: ember_loss(p, a, b, c, force=0.0),
                    {k: v.copy() for k, v in pe.items()},
                    continuous.batches(Xn, Xc, sw, BATCH, np.random.default_rng(10)), STEPS, LR)
        p_u = train(lambda p, a, b, c: ember_loss(p, a, b, c, gate_on=False),
                    {k: v.copy() for k, v in pe.items()},
                    continuous.batches(Xn, Xc, sw, BATCH, np.random.default_rng(11)), STEPS, LR)
        p_g = train(lambda p, a, b, c: ember_loss(p, a, b, c, gate_on=True, lam=0.01),
                    {k: v.copy() for k, v in pe.items()},
                    continuous.batches(Xn, Xc, sw, BATCH, np.random.default_rng(12)), STEPS, LR)

        ms = clean_mse(p_s, Xnt, Xct, force=0.0)
        mu = clean_mse(p_u, Xnt, Xct, gate_on=False)
        mg = clean_mse(p_g, Xnt, Xct, gate_on=True)
        frac, conc = gate_stats(p_g, Xnt, swt)
        helps = mu < ms * 0.97
        keeps = mg <= mu * 1.05
        cheap = frac < 0.5
        concentrates = conc >= 2.0
        verdict = "WIN" if (helps and keeps and cheap and concentrates) else "no"
        print(f"{noise:>6.2f} | {ms:>7.4f} {mu:>7.4f} {mg:>7.4f} | "
              f"{frac:>9.2f} {conc:>5.2f} | {str(helps):>5} {str(keeps):>5} {str(cheap):>5} | "
              f"{verdict}   ({time.time()-t0:.0f}s)")
        rows.append((noise, helps, keeps, cheap, concentrates))

    print("-" * 70)
    wins = [n for n, h, k, c, cc in rows if h and k and c and cc]
    if wins:
        print(f"The economy pays at noise <= {max(wins):.2f}.")
    else:
        print("No clean win at any noise: the arrow helps, and a half-rate gate keeps")
        print("most of the accuracy cheaply — BUT concentration ~1.0x means the gate is")
        print("NOT detecting the switches. Plain prediction gives the gate no reason to")
        print("fire on events, so it degenerates to a constant. The economy needs a task")
        print("that rewards event-detection (as parity did), or explicit gate supervision.")
    print("=" * 70)


if __name__ == "__main__":
    main()
