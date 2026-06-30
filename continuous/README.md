# continuous/ — does the economy pay on a continuous stream?

The headline result Ember verified was the *economy*: a surprise-gate that keeps the
expensive antisymmetric flow dark on predictable steps. The obvious next question, and
the one you asked: does that buy real efficiency on a long, mostly-predictable
continuous signal — the regime closest to live audio/video?

Two experiments here, run on a **switching oscillator**: a noisy signal whose hidden
angular velocity is piecewise-constant and switches rarely, then persists. The model
predicts the *clean* next observation from the *noisy* stream, which needs both
denoising (integrate) and regime-tracking (handle the switches).

```bash
python continuous/efficiency.py        # one detailed operating point + a GRU reference
python continuous/economy_regime.py    # a noise sweep: where (if anywhere) does it pay?
```

## What happened

**The arrow helps.** Turning the antisymmetric flow on every step (uniform) beats the
arrow-off floor (S-only, symmetric recurrence) by ~25–30% on clean-signal MSE. So on
continuous tracking — unlike the parity task — `A` is doing real work.

**But the gate does not learn to fire on the switches.** Across noise levels the gate
concentration on regime changes is ~**1.0×** — i.e. none. In `efficiency.py` the gate
collapses to ~0 (reverts to the S-only floor); in the sweep it settles to a near-constant
~0.5 that keeps most of uniform's accuracy by running the arrow half the time *uniformly*.
Either way it is **not event-driven** — it is a degenerate constant, not the intended
dark-until-surprise behaviour.

| | S-only | uniform | gated | gate conc. | GRU |
|---|---|---|---|---|---|
| efficiency.py (noise 0.30) | 0.0696 | 0.0506 | 0.0674 | 1.02× | 0.0240 |
| sweep (noise 0.00 / 0.10 / 0.30) | rises | best | ≈ uniform @ ~0.5 frac | **1.00×** | — |

## Why — and the recipe it implies

The reason is clean and general: **on a pure prediction task the gate has no reason to
fire on events.** Predicting the clean signal needs the model to *track* omega; it never
needs to know *when* a switch happened. So the optimizer gets no gradient telling it to
align the gate with the switches, and the gate degenerates. Contrast the parity task in
`../experiments/02_kill_keep.py`, where the discrete bit *had* to flip at each event —
there the same gate concentrated **3–20×**, because the task rewarded event-detection.

So the economy is **not an emergent free property of prediction**. It shows up only when
something rewards detecting events. To get event-driven efficiency on real audio/video,
the recipe is one of:

1. **pick an event-rewarding objective** (e.g. predict/flag the change points, or a
   downstream task that depends on them), or
2. **supervise the gate explicitly** — add a term pushing the gate toward the model's own
   normalized prediction-error, so "fire on surprise" is trained, not hoped for.

And separately: a plain GRU still wins on raw accuracy here (0.024 vs 0.051). Ember's
case was never "more accurate" — it was "comparable accuracy at lower *event-time*
compute," and that case is unproven on continuous prediction until the gate is made to
detect events.

## Pointing it at real audio/video

`ember/continuous.py:from_timeseries()` frames any real 1-D or multi-channel signal
(audio RMS/onset envelope, per-frame webcam features, optical-flow vectors, your
Gabor-packet codes) into the same `(n, T, obs_dim)` windows. For real data there is no
clean target, so train self-supervised next-step (predict your own next input); the
honest expectation from the above is that the gate will **not** concentrate on cuts/onsets
unless you add the gate-supervision term in recipe (2). That term is the first thing to
build next.

*Do not hype. Do not lie. Just show.*
