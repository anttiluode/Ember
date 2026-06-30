# Ember

**A sequence cell that holds for free and pays only for surprise.**

PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, June 2026.

> Do not hype. Do not lie. Just show.

---

## What this is

A small, fully-runnable testbed for one idea taken out of the GeometricNeuron line
and turned into an actual trainable architecture: a recurrent cell that factors its
interaction into two halves and **pays for them differently**.

```
C_τ = S ⊕ A
S = (M+Mᵀ)/2   the CHEAP held half — a symmetric associative settle. Always runs.
A = (M−Mᵀ)/2   the DEAR moving half — a skew rotation. Runs ONLY through an
               excitable gate driven by prediction error (surprise).
```

A transformer lives almost entirely in `S` (attention is symmetric associative
retrieval) and staples its arrow on from outside as a positional coordinate. Ember
keeps `A`, **generates** it with an excitable element (you cannot draw an arrow into
passive geometry — V13→V14 in the parent line — you have to break a symmetry to make
one), and only spends compute on it when the input violates what the held state
predicted. Long-horizon memory is parked in a **protected rotor latch** (the whorl
idea): a 2D rotor that flips ~π on a detected surprise and holds for free otherwise.

This repo is not a claim that Ember beats anything. It is an honest **kill/keep**:
build the mechanism, point it at tasks designed to test it, and report what held and
what broke.

---

## The result, in one glance

Run it yourself (numbers below are from the committed code, CPU, seeds fixed):

| claim | status | number |
|---|---|---|
| `A` (skew lag-covariance) carries sequence order | **[V]** | shuffling the time axis collapses ‖A‖ **5.05×**, top island **4.85×** |
| the surprise-gate makes arrow-compute track surprise | **[V]** | gate fires **3.4× more** on surprises than quiet; arrow **dark ~75–90%** of steps |
| the gate is load-bearing (not decoration) | **[V]** | removing it (gate≡1) drops parity from 0.63 → 0.53 (chance) |
| the cell solves hard long-horizon discrete memory | **[K]** | Ember **0.63** vs a parameter-matched **GRU 0.91** on 64-step parity |
| the protected latch reliably stores a bit | **[B/K]** | works as a detector; brittle to residual false-flips over long horizons |

**The honest headline:** the *economy* is real — a structural surprise-gate that
keeps the expensive directional flow dark until prediction breaks, demonstrably and
load-bearingly. The *topological memory latch* is built and partly works but is
beaten by a plain recurrent state on the one task that stresses it — the same wall
the parent line already hit ("a phase wheel interpolates; fine for heading, nonsense
for the digit 7").

---

## Run the two experiments

```bash
pip install -r requirements.txt
python experiments/01_probe_sanity.py    # does A actually carry order? (falsification)
python experiments/02_kill_keep.py       # train Ember vs GRU; does the gate earn its keep?
```

Each prints its own numbers and its own verdict. Nothing is hidden in a figure the
print-out doesn't also state.

---

## The map — every file, what it is, its status

Status legend: **[V]** verified in code · **[K]** a claim this killed · **[B]** still a bet · **[LIVE]** a runnable instrument.

### `ember/` — the engine
| file | what it is | status |
|---|---|---|
| `cell.py` | the Ember cell: symmetric settle `S`, skew flow `A`, excitable surprise gate with refractory, protected rotor latch, two readout heads. The whole unit. | **[V]** runs / **[K]** as a GRU-beater |
| `probe.py` | the `S ⊕ A` diagnostic: split the lag covariance of any hidden states, return ‖A‖, the imaginary spectrum (islands), and the shuffle falsification. Includes a hook to run on a real pretrained transformer. | **[V][LIVE]** |
| `baselines.py` | a parameter-matched GRU (the external reference) + the gate-ablated control. | **[V]** |
| `data.py` | the two synthetic tasks: smooth carrier (which the `S`-half alone solves) and parity-of-surprises (which needs detect→flip→hold). | **[V]** |
| `optim.py` | minimal Adam over a dict of params. | **[V]** |

### `experiments/`
| file | what it is | status |
|---|---|---|
| `01_probe_sanity.py` | falsifies the premise on synthetic ground truth: a directed signal vs an order-free one, and the same signal time-shuffled. The arrow must collapse when order dies. | **[V]** premise holds (5.05×) |
| `02_kill_keep.py` | trains Ember, the gate-ablated control, and a GRU on parity-of-surprises; reports accuracy AND whether the gate concentrates on surprise. | **[V/K]** mixed, reported honestly |

---

## The ledger, consolidated

**Verified in code:**
- the `S ⊕ A` premise — `A` carries order; destroying time order collapses ‖A‖ ~5× (`01`);
- the economy — a structural, refractory surprise-gate keeps the directional flow dark
  on predictable steps and fires it on surprise (mean gate ~0.1–0.25; concentration 3–20×
  depending on how hard the latch fights the detector) (`02`);
- the gate is load-bearing — ablating it (gate≡1) collapses the memory task to chance (`02`).

**Killed by the builds (the useful negatives):**
- "the surprise-gate concentrates on a smooth next-step task" — **false**: the cheap `S`-half
  already solves smooth prediction, so the arrow is never needed and the gate has nothing to
  specialize on. You cannot test an economy on a task the symmetric half solves for free.
- "an unbounded winding read by a linear head stores a discrete count" — **false**: variable
  per-event phase increment + linear readout cannot do parity. Same failure as the parent
  line's phase wheel.
- "the cell, end-to-end, beats a recurrent baseline at long-horizon discrete memory" —
  **false here**: GRU 0.91 vs Ember 0.63 on 64-step parity. The pieces work in isolation;
  composed and trained jointly, the residual false-flip rate corrupts parity over a long horizon.

**Still a bet:**
- that the protected rotor latch can be made robust enough (sharper detector, hard commitment,
  or a discrete straight-through flip) to match a recurrent state on long-horizon memory;
- that the real payoff regime is continuous-time / event-stream / very-long-context data
  rather than the short synthetic tasks here — untested, needs real data;
- that the probe's `A`-concentration prediction holds on a **real** trained transformer
  (`probe.harvest_hidden_states` is wired but needs a model download — run it where you have
  the network).

---

## What would actually move this

1. **Run the probe on a real model.** `ember/probe.py` has the hook. Does `‖A‖` spike on
   structured input (code, math) and go flat on scrambled tokens, and peak in middle layers?
   That is the cheap experiment that licenses everything else.
2. **Fix the latch or kill it.** Replace the soft rotor with a hard straight-through flip and
   see if parity reaches GRU level. If it can't be made robust, the honest move is to drop the
   topological-memory claim and keep only the economy.
3. **Find the regime where the economy pays.** The gate's value is dark-on-predictable. That
   only matters when most of the stream is predictable and compute is the constraint — long
   continuous-time signals, not 64-step toys.

---

## Lineage

Built on the GeometricNeuron / GAIT / PerceptionLab line (`github.com/anttiluode`),
specifically the V13→V14 result that direction must be *generated* by an active element,
the whorl's topological latch, and `the_tensor`'s spend-on-surprise economy. The original
insight and direction are Antti Luode's; this build, its experiments, and this ledger were
written with Claude (Opus 4.8).

*Do not hype. Do not lie. Just show.*
