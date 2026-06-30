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
| the gate is load-bearing (not decoration) | **[V]** | removing it (gate≡1) collapses parity to chance |
| the cell solves hard long-horizon discrete memory | **[K]** | parameter-matched **GRU 0.91** vs **Ember 0.53–0.71** on 64-step parity (seed-dependent) |
| making the rotor flip **discrete** fixes the latch | **[K]** | hard straight-through flip helps **+0.07** but **kills the economy** (concentration 3.4×→**1.0×**) and still loses to GRU — *see exp 03* |
| the protected latch reliably stores a bit | **[B/K]** | works as a detector; brittle to residual false-flips over long horizons |
| the arrow helps on a continuous tracking stream | **[V]** | arrow-on (uniform) beats arrow-off (S-only) by **~25–30%** clean-MSE |
| the gate concentrates compute on a continuous *prediction* task | **[K]** | concentration **~1.0×** — plain prediction gives no reason to fire on events |

**The honest headline:** the *economy* is real **but conditional** — the surprise-gate
keeps the expensive flow dark and is load-bearing, yet it only concentrates on events
when the *task rewards detecting events*. The *topological memory latch* is built and
partly works but is beaten by a plain recurrent state on the one task that stresses it.
The newest result (exp 03) makes the diagnosis sharper, not softer: **the soft latch is
not brittle just because it is soft.** Hardening the flip into a discrete straight-through
commit *does* nudge parity up, but it does so by abandoning the very thing that makes Ember
Ember — the gate stops concentrating on surprise (3.4×→1.0×) and the task head simply routes
around the latch. Discreteness is not the missing piece. **The latch is still the open seam,
and the fix is upstream of it — in the gate.**

---

## Run the experiments

```bash
pip install -r requirements.txt
python experiments/01_probe_sanity.py    # does A actually carry order? (falsification)
python experiments/02_kill_keep.py       # train Ember vs GRU; does the gate earn its keep?
python experiments/03_fix_the_latch.py   # soft vs HARD straight-through latch vs GRU  [NEW]
python continuous/efficiency.py          # does the economy pay on a continuous stream?
python continuous/economy_regime.py      # noise sweep: where (if anywhere) does it pay?
```

See [`continuous/README.md`](continuous/README.md) for the efficiency story in full.

Each prints its own numbers and its own verdict. Nothing is hidden in a figure the
print-out doesn't also state.

---

## Experiment 03 — fix the latch, in full

The parent line and §3 below pointed at one move: *replace the soft rotor with a hard
straight-through flip and see if parity reaches GRU level.* `cell_hard.py` is exactly
`cell.py` with one change — the rotor commitment is a Bengio straight-through estimator
(forward: a full π flip or nothing; backward: the soft sigmoid gradient, so the detector
still trains). `03_fix_the_latch.py` runs it head-to-head against the soft cell and the
GRU on the same parity-of-surprises task, parameter-matched, then checks the gap across
three seeds and measures rotor drift on quiet steps.

**Headline (900 steps, params Ember=5108 / GRU=5097):**

| variant | parity acc | gate concentration | quiet-step rotor drift |
|---|---|---|---|
| Ember-soft | 0.714 | **3.41×** | 0.337 rad/step |
| Ember-hard | **0.783** | **1.00×** | 0.638 rad/step |
| GRU | 0.875 | — | — |

soft→hard Δacc = **+0.069**, hard vs GRU = **−0.091**.

**3-seed check (450 steps each):**

| | soft | hard | GRU |
|---|---|---|---|
| mean ± sd | 0.532 ± 0.013 | **0.587 ± 0.040** | 0.912 ± 0.033 |

**The three diagnostics, and what each killed or kept:**

- **[V]** the hard flip *does* improve parity over the soft latch — small but consistent
  across seeds (+0.07 at full budget, +0.055 mean at the short budget). Discreteness is
  not nothing.
- **[K]** the hard flip does **not** remove quiet-step rotor drift — it *raises* it
  (0.337 → 0.638 rad/step). The discrete commit fires full-π flips on quiet steps too,
  because nothing is stopping it.
- **[K]** the hard flip does **not** reach GRU level (0.78 vs 0.88 at 900 steps; 0.59 vs
  0.91 across seeds), and it gets even that far by **killing the economy**: gate
  concentration collapses from 3.41× to 1.00×. The flip became cheap and generally useful,
  so training drove the gate everywhere and the linear task head learned to read the result
  without relying on a *selective* latch. It routed around the seam instead of closing it.

**Verdict: PARTIAL.** Discreteness helps a little; it is not the fix. The economy
(dark-on-predictable) and the accuracy are now visibly in tension under a pure parity loss,
and the latch is still the open seam. The corollary points straight at the next move: the
gate has to be *made* to stay concentrated, by supervision, before any latch refinement can
hold its meaning.

---

## The map — every file, what it is, its status

Status legend: **[V]** verified in code · **[K]** a claim this killed · **[B]** still a bet · **[LIVE]** a runnable instrument.

### `ember/` — the engine
| file | what it is | status |
|---|---|---|
| `cell.py` | the Ember cell: symmetric settle `S`, skew flow `A`, excitable surprise gate with refractory, protected rotor latch (soft commit), two readout heads. The whole unit. | **[V]** runs / **[K]** as a GRU-beater |
| `cell_hard.py` | identical cell with one change: the rotor commit is a **hard straight-through flip** (forward π-or-nothing, soft gradient). The thing exp 03 tests. | **[V]** runs / **[K]** as a latch fix |
| `probe.py` | the `S ⊕ A` diagnostic: split the lag covariance of any hidden states, return ‖A‖, the imaginary spectrum (islands), and the shuffle falsification. Includes a hook to run on a real pretrained transformer. | **[V][LIVE]** |
| `baselines.py` | a parameter-matched GRU (the external reference) + the gate-ablated control. | **[V]** |
| `data.py` | the two synthetic tasks: smooth carrier (which the `S`-half alone solves) and parity-of-surprises (which needs detect→flip→hold). | **[V]** |
| `optim.py` | minimal Adam over a dict of params. | **[V]** |

### `experiments/`
| file | what it is | status |
|---|---|---|
| `01_probe_sanity.py` | falsifies the premise on synthetic ground truth: a directed signal vs an order-free one, and the same signal time-shuffled. The arrow must collapse when order dies. | **[V]** premise holds (5.05×) |
| `02_kill_keep.py` | trains Ember, the gate-ablated control, and a GRU on parity-of-surprises; reports accuracy AND whether the gate concentrates on surprise. | **[V/K]** mixed, reported honestly |
| `03_fix_the_latch.py` | soft latch vs hard straight-through latch vs GRU on parity; 3-seed gap check + quiet-step rotor-drift diagnostic. | **[K]** discreteness helps but doesn't close it; economy collapses |

### `continuous/` — does the economy pay on a stream? (see its own README)
| file | what it is | status |
|---|---|---|
| `efficiency.py` | one operating point on a noisy switching-oscillator: S-only floor vs uniform-arrow vs gated vs GRU. | **[V]** arrow helps / **[K]** gate doesn't concentrate |
| `economy_regime.py` | a noise sweep mapping where event-driven gating would pay. | **[K]** ~1.0× concentration everywhere |
| `continuous.py` | the switching-oscillator generator + `from_timeseries()` adapter for real audio/video. | **[V][LIVE]** |

---

## The ledger, consolidated

**Verified in code:**
- the `S ⊕ A` premise — `A` carries order; destroying time order collapses ‖A‖ ~5× (`01`);
- the economy — a structural, refractory surprise-gate keeps the directional flow dark on
  predictable steps and fires it on surprise (concentration 3–20× on the task that rewards
  detecting events) (`02`);
- the gate is load-bearing — ablating it (gate≡1) collapses the memory task to chance (`02`);
- discreteness gives a small real bump — the hard straight-through flip beats the soft latch
  on parity, consistently across seeds (`03`).

**Killed by the builds (the useful negatives):**
- "the surprise-gate concentrates on a smooth next-step task" — **false**: the cheap `S`-half
  already solves smooth prediction, so the arrow is never needed. You cannot test an economy on
  a task the symmetric half solves for free.
- "an unbounded winding read by a linear head stores a discrete count" — **false**: variable
  per-event phase increment + linear readout cannot do parity. Same failure as the parent line's
  phase wheel.
- "the cell, end-to-end, beats a recurrent baseline at long-horizon discrete memory" —
  **false here**: GRU ~0.91 vs Ember 0.53–0.71 on 64-step parity.
- "a hard, discrete flip fixes the latch" — **false** (`03`): it helps parity a little but
  *raises* quiet-step drift (0.34→0.64 rad/step), **collapses gate concentration 3.4×→1.0×**,
  and still loses to the GRU. The accuracy it buys comes from the task head routing around a
  no-longer-selective latch — not from the latch holding. Discreteness is not the missing piece.
- "the surprise-gate spontaneously concentrates on events during continuous prediction" —
  **false**: ~1.0× concentration across noise levels. A pure prediction objective never needs to
  know *when* a change happened.

**Still a bet:**
- that **supervising the gate** — pushing it toward the model's own normalized prediction-error
  so "fire on surprise" is trained rather than hoped for — is what keeps concentration high, and
  is the prerequisite that exp 03 just showed is missing. This is now the clear next experiment,
  not the latch.
- that, *with* a supervised gate holding concentration, a hard straight-through latch then has a
  selective signal to latch onto and can be made to hold over a long horizon.
- that the probe's `A`-concentration prediction holds on a **real** trained transformer
  (`probe.harvest_hidden_states` is wired but needs a model download).

---

## What would actually move this

Exp 03 reordered this list. The latch was the obvious suspect; the run says the gate is the
real one.

1. **Supervise the gate (now the clear next move).** Exp 03 showed that under a pure parity/
   prediction loss the gate will *trade away* its concentration to gain accuracy. Add a term
   pushing the gate toward the model's own normalized prediction-error, re-run `03` and
   `continuous/economy_regime.py`, and see whether concentration stays high *while* parity
   improves. This is the prerequisite for both the latch and any real-stream efficiency claim.
2. **Then, and only then, re-test the hard latch.** With a supervised, selective gate, the
   straight-through flip from `cell_hard.py` has something meaningful to fire on. Re-run `03`
   and see if hard + supervised closes the GRU gap that hard alone could not.
3. **Run the probe on a real model.** `probe.py` has the hook. Does `‖A‖` spike on structured
   input (code, math) and go flat on scrambled tokens, and peak in middle layers?
4. **Then point it at real audio/video.** `continuous.py:from_timeseries()` frames any real
   stream into the harness — but only after (1), or the gate will sit at a constant.

---

## Lineage

Built on the GeometricNeuron / GAIT / PerceptionLab line (`github.com/anttiluode`),
specifically the V13→V14 result that direction must be *generated* by an active element,
the whorl's topological latch, and `the_tensor`'s spend-on-surprise economy. The original
insight and direction are Antti Luode's; this build, its experiments, and this ledger were
written with Claude (Opus 4.8).

*Do not hype. Do not lie. Just show.*
