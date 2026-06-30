"""
Sparse-surprise sequence task.

A predictable carrier (a phase advancing at a fixed rate, observed through a few
harmonics) with RARE phase resets. Between resets the next observation is almost
perfectly predictable from the last; at a reset it is not. This is the regime a
transformer is worst at and the regime Ember is built for: long stretches of
free holding punctuated by isolated moments that actually need the arrow.

generate() returns:
    X     (n, T, obs_dim)  the observations
    jumps (n, T)  bool, True at the step whose value was made unpredictable by a reset

The `jumps` mask is GROUND TRUTH surprise. No model sees it. It is only used,
after training, to ask: did the gate light up where the surprise actually was?
"""
import numpy as np


def generate(n_seqs=512, T=64, obs_dim=4, omega=0.35, p_jump=0.06,
             jump_scale=2.5, seed=0):
    rng = np.random.default_rng(seed)
    n_harm = max(1, obs_dim // 2)
    X = np.zeros((n_seqs, T, obs_dim), dtype=np.float64)
    jumps = np.zeros((n_seqs, T), dtype=bool)

    for i in range(n_seqs):
        phi = rng.uniform(0, 2 * np.pi)
        for t in range(T):
            # rare phase reset
            if t > 0 and rng.random() < p_jump:
                phi = phi + rng.normal() * jump_scale
                jumps[i, t] = True
            for h in range(n_harm):
                X[i, t, 2 * h] = np.cos((h + 1) * phi)
                if 2 * h + 1 < obs_dim:
                    X[i, t, 2 * h + 1] = np.sin((h + 1) * phi)
            phi = phi + omega
    return X, jumps


def batches(X, jumps, batch_size, rng):
    """Yield random mini-batches forever."""
    n = X.shape[0]
    while True:
        idx = rng.integers(0, n, size=batch_size)
        yield X[idx], jumps[idx]


def generate_parity(n_seqs=512, T=64, obs_dim=4, omega=0.35, p_jump=0.06,
                    seed=0):
    """
    Protected-memory task. A rotating carrier with RARE, detectable phase resets.
    The target at each step is the running PARITY (count mod 2) of resets seen so
    far. Solving it requires: detect each reset (a surprise), flip one protected
    bit, and hold it for free until the next reset. This is the regime the Ember
    cell is built for, and the regime the smooth next-step task could not test.

    Returns:
        X     (n, T, obs_dim)  the carrier
        Y     (n, T, 1)        running parity target in {0,1}
        jumps (n, T)  bool, ground-truth reset locations
    """
    rng = np.random.default_rng(seed)
    n_harm = max(1, obs_dim // 2)
    X = np.zeros((n_seqs, T, obs_dim))
    Y = np.zeros((n_seqs, T, 1))
    jumps = np.zeros((n_seqs, T), dtype=bool)
    for i in range(n_seqs):
        phi = rng.uniform(0, 2 * np.pi)
        count = 0
        for t in range(T):
            if t > 0 and rng.random() < p_jump:
                phi = phi + rng.uniform(np.pi / 2, 3 * np.pi / 2)  # big, detectable
                count += 1
                jumps[i, t] = True
            for h in range(n_harm):
                X[i, t, 2 * h] = np.cos((h + 1) * phi)
                if 2 * h + 1 < obs_dim:
                    X[i, t, 2 * h + 1] = np.sin((h + 1) * phi)
            Y[i, t, 0] = count % 2
            phi = phi + omega
    return X, Y, jumps


def batches_xy(X, Y, jumps, batch_size, rng):
    n = X.shape[0]
    while True:
        idx = rng.integers(0, n, size=batch_size)
        yield X[idx], Y[idx], jumps[idx]
