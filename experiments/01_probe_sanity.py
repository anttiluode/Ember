"""
Experiment 01 — does the skew half A actually carry the order?

The premise of the whole Ember design is that direction/sequence lives in A, the
skew half of the lagged covariance, while S is order-blind. Before building on
that, falsify it: if it is true, destroying the time order must collapse the
arrow_ratio, and a directed (rotating) signal must show a far bigger arrow than
an order-free one.

This runs offline on synthetic signals where we KNOW the ground truth. To run the
same probe on a real pretrained transformer, use ember.probe.harvest_hidden_states
(needs network + transformers); the code is in probe.py.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from ember.probe import s_a_decompose, run_falsification


def rotating_signal(T=400, D=8, n_modes=3, seed=0):
    """A directed signal: several 2D planes each rotating one way. Pure arrow."""
    rng = np.random.default_rng(seed)
    H = np.zeros((T, D))
    t = np.arange(T)
    for m in range(n_modes):
        w = 0.1 + 0.25 * m
        a, b = 2 * m % D, (2 * m + 1) % D
        ph = rng.uniform(0, 2 * np.pi)
        H[:, a] += np.cos(w * t + ph)
        H[:, b] += np.sin(w * t + ph)   # +sin => consistent rotation sense
    H += 0.05 * rng.normal(size=H.shape)
    return H


def standing_signal(T=400, D=8, seed=0):
    """An order-free control: every channel is an INDEPENDENT cosine. The lag
    covariance of even (cosine) signals is symmetric in expectation, so A should
    sit near the noise floor — no consistent rotation, no arrow."""
    rng = np.random.default_rng(seed)
    H = np.zeros((T, D))
    t = np.arange(T)
    for c in range(D):
        w = 0.1 + 0.25 * rng.random()
        H[:, c] = np.cos(w * t + rng.uniform(0, 2 * np.pi))
    H += 0.05 * rng.normal(size=H.shape)
    return H


def main():
    print("=" * 64)
    print("EXPERIMENT 01  —  probe falsification (synthetic ground truth)")
    print("=" * 64)

    rot = rotating_signal()
    stand = standing_signal()

    r_rot = s_a_decompose(rot)
    r_stand = s_a_decompose(stand)

    print("\nA-arrow on a DIRECTED (rotating) signal vs an ORDER-FREE (standing) one")
    print(f"  rotating  ||A|| = {r_rot['a_norm']:.4f}  top island = {r_rot['top_island']:.4f}  "
          f"islands = {np.round(r_rot['islands'][:3], 3)}")
    print(f"  standing  ||A|| = {r_stand['a_norm']:.4f}  top island = {r_stand['top_island']:.4f}  "
          f"islands = {np.round(r_stand['islands'][:3], 3)}")
    print(f"  --> directed/order-free ||A|| ratio = "
          f"{r_rot['a_norm'] / (r_stand['a_norm'] + 1e-12):.2f}x")

    print("\nShuffle the time axis of the directed signal (destroys order):")
    f = run_falsification(rot, n_shuffle=30)
    print(f"  ordered   ||A|| = {f['ordered_anorm']:.4f}   top island = {f['ordered_island']:.4f}")
    print(f"  shuffled  ||A|| = {f['shuffled_anorm']:.4f}   top island = {f['shuffled_island']:.4f}")
    print(f"  --> ordered/shuffled ||A|| = {f['anorm_ratio']:.2f}x   "
          f"top island = {f['island_ratio']:.2f}x")

    keep = (f['anorm_ratio'] > 2.0) and (f['island_ratio'] > 2.0)
    print("\nVERDICT:", "PREMISE HOLDS — A carries the order; shuffling collapses it."
          if keep else "PREMISE WEAK — A did not separate order from disorder.")
    print("(decisive test = the shuffle collapse; directed/standing is supporting context)")
    print("=" * 64)


if __name__ == "__main__":
    main()
