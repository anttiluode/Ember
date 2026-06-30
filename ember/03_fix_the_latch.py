"""
03 — fix-the-latch-or-kill-it  (README move #3, PAPER §4 open seam)

Same parity-of-surprises task and the same parameter-matched setup as 02, but
adds the HARD straight-through rotor (ember/cell_hard.py) as a third arm:

    Ember-soft   the committed cell — soft commit = sigmoid(8*(g-0.5))
    Ember-hard   identical cell, rotor flip made discrete via straight-through
    GRU          the external reference

Question the ledger poses: the detector is fine; is the *latch* what is brittle?
If so, a discrete flip should close the 0.63 -> 0.91 gap. If it does not, the
topological-memory claim should be dropped and only the economy kept.

Also reports a ROTOR-DRIFT diagnostic: on quiet (non-surprise) steps the rotor
should not move. We measure the mean per-step angular wobble of the rotor on
quiet steps for each variant — the soft cell's accumulated sub-pi drift is the
hypothesised failure mechanism, and this number makes it visible.

Single training budget matched to 02 (900 steps). A 3-seed check on the two
Ember arms follows, to show the soft/hard gap is not one lucky seed.

Do not hype. Do not lie. Just show.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from autograd import value_and_grad

from ember import cell, cell_hard, baselines, data
from ember.optim import Adam, count_params


def sigmoid_np(x):
    return 0.5 * (np.tanh(0.5 * x) + 1.0)


def train(loss_fn, params, gen, steps, lr, label=""):
    opt = Adam(params, lr=lr)
    vg = value_and_grad(loss_fn)
    t0 = time.time()
    for i in range(steps):
        X, Y, _J = next(gen)
        L, g = vg(params, X, Y)
        params = opt.step(params, g)
        if (i + 1) % max(1, steps // 4) == 0:
            print(f"    [{label}] {i+1:4d}/{steps}  loss={float(L):.5f}", flush=True)
    print(f"    [{label}] {time.time()-t0:.1f}s", flush=True)
    return params


def parity_acc(mod, p, X, Y, gate_on=True, warmup=4):
    out = mod.forward(p, X, gate_on=gate_on)
    pred = (sigmoid_np(np.asarray(out["task"])) > 0.5).astype(float)
    m = np.zeros_like(Y, dtype=bool); m[:, warmup:] = True
    return float((pred[m] == Y[m]).mean())


def gru_acc(p, X, Y, warmup=4):
    out = baselines.forward(p, X)
    pred = (sigmoid_np(np.asarray(out["task"])) > 0.5).astype(float)
    m = np.zeros_like(Y, dtype=bool); m[:, warmup:] = True
    return float((pred[m] == Y[m]).mean())


def economy(mod, p, X, jumps, warmup=4):
    out = mod.forward(p, X, gate_on=True)
    gate = np.asarray(out["gate"])
    m = np.zeros_like(jumps); m[:, warmup:] = True
    gj = gate[jumps & m].mean(); gn = gate[(~jumps) & m].mean()
    return float(gj / (gn + 1e-9)), float(gate[m].mean())


def rotor_quiet_drift(mod, p, X, jumps, warmup=4):
    """Mean per-step rotor angular movement on QUIET steps (should be ~0).
    Reconstructs the rotor exactly as the cell does, recording the flip angle
    actually applied each step, then averages |angle| over non-surprise steps."""
    out = mod.forward(p, X, gate_on=True)
    g = np.asarray(out["gate"])                       # (B,T)
    hard = mod is cell_hard
    if hard:
        commit = (g > 0.5).astype(float)
    else:
        commit = sigmoid_np(8.0 * (g - 0.5))
    ang = np.pi * commit                              # (B,T) applied flip angle
    m = np.zeros_like(jumps); m[:, warmup:] = True
    quiet = (~jumps) & m
    return float(np.abs(ang[quiet]).mean())


def one_run(seed_init, seed_data, steps=900, verbose=True):
    obs_dim, T = 4, 64
    Xtr, Ytr, Jtr = data.generate_parity(n_seqs=640, T=T, obs_dim=obs_dim,
                                          p_jump=0.06, seed=seed_data)
    Xte, Yte, Jte = data.generate_parity(n_seqs=256, T=T, obs_dim=obs_dim,
                                          p_jump=0.06, seed=seed_data + 100)
    BATCH, LR = 32, 4e-3
    pe = cell.init_params(obs_dim=obs_dim, D=48, K=4, seed=seed_init)
    pg = baselines.init_params(obs_dim=obs_dim, H=38, seed=seed_init)

    gen_s = data.batches_xy(Xtr, Ytr, Jtr, BATCH, np.random.default_rng(seed_data))
    gen_h = data.batches_xy(Xtr, Ytr, Jtr, BATCH, np.random.default_rng(seed_data))
    gen_g = data.batches_xy(Xtr, Ytr, Jtr, BATCH, np.random.default_rng(seed_data + 1))

    if verbose: print("\n-- Ember-soft --", flush=True)
    ps = train(lambda p, X, Y: cell.parity_loss(p, X, Y, gate_on=True, lam=0.02),
               {k: v.copy() for k, v in pe.items()}, gen_s, steps, LR, "soft")
    if verbose: print("\n-- Ember-hard (straight-through) --", flush=True)
    ph = train(lambda p, X, Y: cell_hard.parity_loss(p, X, Y, gate_on=True, lam=0.02),
               {k: v.copy() for k, v in pe.items()}, gen_h, steps, LR, "hard")
    if verbose: print("\n-- GRU --", flush=True)
    pgt = train(lambda p, X, Y: baselines.parity_loss(p, X, Y),
                pg, gen_g, steps, LR, "gru")

    res = {
        "acc_soft": parity_acc(cell, ps, Xte, Yte),
        "acc_hard": parity_acc(cell_hard, ph, Xte, Yte),
        "acc_gru":  gru_acc(pgt, Xte, Yte),
        "ratio_soft": economy(cell, ps, Xte, Jte)[0],
        "ratio_hard": economy(cell_hard, ph, Xte, Jte)[0],
        "drift_soft": rotor_quiet_drift(cell, ps, Xte, Jte),
        "drift_hard": rotor_quiet_drift(cell_hard, ph, Xte, Jte),
        "params_e": count_params(pe), "params_g": count_params(pg),
    }
    return res


def main():
    print("=" * 66)
    print("EXPERIMENT 03  —  fix the latch: hard straight-through rotor")
    print("=" * 66)
    r = one_run(seed_init=0, seed_data=1, steps=900)
    print("\n" + "-" * 66)
    print(f"HEADLINE  (params: Ember={r['params_e']}  GRU={r['params_g']})")
    print(f"  Ember-soft  parity acc = {r['acc_soft']:.3f}   "
          f"gate concentration = {r['ratio_soft']:.2f}x   "
          f"quiet-drift = {r['drift_soft']:.4f} rad/step")
    print(f"  Ember-hard  parity acc = {r['acc_hard']:.3f}   "
          f"gate concentration = {r['ratio_hard']:.2f}x   "
          f"quiet-drift = {r['drift_hard']:.4f} rad/step")
    print(f"  GRU         parity acc = {r['acc_gru']:.3f}")
    print(f"\n  soft->hard Δacc = {r['acc_hard']-r['acc_soft']:+.3f}   "
          f"hard vs GRU gap = {r['acc_hard']-r['acc_gru']:+.3f}")

    print("\n" + "-" * 66)
    print("3-SEED CHECK on the soft/hard gap (450 steps each, faster)")
    accs_s, accs_h, accs_g = [], [], []
    for sd in (11, 22, 33):
        rr = one_run(seed_init=sd, seed_data=sd, steps=450, verbose=False)
        accs_s.append(rr["acc_soft"]); accs_h.append(rr["acc_hard"]); accs_g.append(rr["acc_gru"])
        print(f"  seed {sd:2d}:  soft={rr['acc_soft']:.3f}  hard={rr['acc_hard']:.3f}  "
              f"gru={rr['acc_gru']:.3f}", flush=True)
    import statistics as st
    print(f"\n  soft  {st.mean(accs_s):.3f} ± {st.pstdev(accs_s):.3f}")
    print(f"  hard  {st.mean(accs_h):.3f} ± {st.pstdev(accs_h):.3f}")
    print(f"  gru   {st.mean(accs_g):.3f} ± {st.pstdev(accs_g):.3f}")

    print("\n" + "-" * 66)
    print("LEDGER")
    closed = r["acc_hard"] >= 0.90
    helped = (r["acc_hard"] - r["acc_soft"]) >= 0.05
    drift_killed = r["drift_hard"] <= 0.5 * r["drift_soft"] + 1e-9
    print(f"  [{'V' if drift_killed else 'K'}] hard flip removes quiet-step rotor drift: "
          f"{drift_killed} (soft {r['drift_soft']:.4f} -> hard {r['drift_hard']:.4f} rad/step)")
    print(f"  [{'V' if helped else 'K'}] hard flip improves parity over soft (Δ>=0.05): {helped}")
    print(f"  [{'V' if closed else 'K'}] hard flip reaches GRU level (acc>=0.90): {closed}")
    if closed:
        v = "KEEP — the latch was the problem; a discrete flip closes it."
    elif helped:
        v = "PARTIAL — discreteness helps but does not fully close; latch still the seam."
    else:
        v = "KILL the latch — discreteness does not rescue it; keep only the economy."
    print(f"\n  VERDICT: {v}")
    print("=" * 66)


if __name__ == "__main__":
    main()
