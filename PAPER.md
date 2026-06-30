# Ember: an arrow you have to pay for

### Turning one organ of the GeometricNeuron line — the priced, generated arrow — into a trainable sequence cell, and reporting what it can and cannot do.

**PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, June 2026.**

> Do not hype. Do not lie. Just show.

---

## 0. The one move

A modern transformer lives in the symmetric half of the time-lagged covariance.
Attention is `QKᵀ` — a content overlap, an associative store, the modern-Hopfield
energy. It is order-blind by construction; everything about *sequence* is added from
outside as a coordinate (positional encoding, a causal mask). In the language of the
parent line, the transformer is `C_τ = S ⊕ A` with `A` amputated and a fake arrow
stapled back on. Time is a label, not a flow.

Ember is the smallest honest attempt to keep `A` and make it earn its place. The cell
factors its own interaction into the two halves and prices them differently:

- `S`, the symmetric half, is a **cheap associative settle** — it holds content, it
  runs every step, it is the part that already works in every model.
- `A`, the skew half, is the **arrow** — a rotation through sequence-space — and it is
  expensive, so it runs only through an **excitable gate** that opens on prediction
  error and is suppressed by a refractory trace.

The gate is the load-bearing idea. It is the FitzHugh-Nagumo correction from the parent
line (V13→V14): you cannot *draw* an arrow into a passive linear medium — a passive
medium is reciprocal — you have to *break a symmetry* with an active, history-dependent
element to *make* one. The refractory trace is what makes the firing history-dependent,
which is what makes the flow non-reciprocal rather than a reversible linear rotation.
And because the gate is driven by surprise, the expensive half is dark whenever the
world is predictable and lit only when it is not. That is `the_tensor`'s economy —
spend compute in proportion to prediction error — turned from a measurement into a
layer rule.

Long-horizon memory is parked in a **protected rotor**: a 2D latch that flips ~π when a
surprise is detected and holds otherwise. This is the whorl idea — a winding is an
integer charge, it does not continuously drift — pressed into service as a flip-flop.

---

## 1. What we tested, and the trap we fell into first

The first task was the obvious one: predict the next step of a mostly-predictable
carrier with rare disturbances. It failed to test anything — and the failure was
instructive. On smooth next-step prediction the cheap `S`-half **already solves the
task**, so the arrow is never needed, the gate-ablated control gets the identical
score, and the gate has nothing to specialise on (it sat at ~0.5 everywhere,
concentration 1.08×). *You cannot test an economy on a task the symmetric half solves
for free.* This is logged as a kill, because it is the kind of thing that looks like a
result until you read it.

The task that actually exercises the mechanism is **parity of surprises**: a rotating
carrier with rare, detectable phase resets, where the target is the running count of
resets mod 2. Solving it requires the full pipeline — detect each reset (fire the gate),
flip the protected bit (turn the rotor), and hold it for free until the next one. A
continuous recurrent state has to maintain a flip-flop across the whole sequence; Ember
should, in principle, fire only at the resets.

---

## 2. What held

Two things held, cleanly.

**The premise.** The skew half `A` of the lag covariance genuinely carries order.
Shuffle the time axis of a directed signal and ‖A‖ collapses 5× and its top island
4.85×, while an order-free control sits near the noise floor. The thing Ember is built
to read is really there to be read.

**The economy.** The surprise-gate works as designed: it fires several times more often
on the ground-truth reset steps than on quiet steps and leaves the arrow dark the large
majority of the time. And it is **load-bearing**, not decorative: force the gate to 1
(pay the arrow every step) and the protected rotor flips on every step, the memory
scrambles, and the task collapses to chance. The gated cell beats its own ungated
ablation. The arrow being *off* is doing real work.

---

## 3. What broke

The cell does **not** match a plain GRU on the hard version of the task. Parameter-
matched, on 64-step parity, the GRU reaches ~0.91 and Ember ~0.63 — above chance, above
its ablation, but well short. The reason is specific and worth stating because it is the
same wall the parent line already documented: parity over a long horizon is unforgiving,
a single missed or spurious flip inverts every label that follows, and the residual
false-flip rate of even a fairly sharp gate is enough to corrupt the count over 64 steps.
The detector is good; the *latch built on top of the detector* is brittle, and a smooth
recurrent state simply does the bookkeeping better. An unbounded winding read by a linear
head was worse still — it cannot represent count-mod-2 at all, exactly the "phase wheel
stores nonsense for the digit 7" failure from the Sudoku arc.

So the composition is the problem. Each organ works in isolation — the skew operator
carries order, the gate detects surprise, a rotor can hold a bit — but trained jointly
and pushed to do long-horizon discrete memory, the assembly is beaten by a baseline that
has none of the structure and just learns a clean recurrence.

---

## 4. The honest reading

Strip it down and this is what Ember is, at this point:

- a **demonstrated** mechanism — a structural, refractory surprise-gate that makes a
  directional computation's cost track prediction error, dark when the world is
  predictable, lit when it is not, and load-bearing in that removing it breaks the task;
- a **demonstrated** premise underneath it — the skew half of the lag covariance carries
  the order, and loses it when the order is destroyed;
- a **failed** ambition, reported as such — that this composes, end-to-end, into a unit
  that beats a recurrent baseline at long-horizon discrete memory. It does not, here.

That is a smaller claim than "a new kind of AI" and a real one. The value Ember actually
shows is the **economy** — compute that goes dark on the predictable and lights up on the
surprising — and that value is most likely to matter exactly where the toy tasks here do
not reach: long continuous-time or event-stream data where most of the signal is
predictable and the constraint is compute, not capacity. The protected-memory latch is,
honestly, still a bet, and the next move is to make it a hard straight-through flip and
see if it can be made to hold — or to drop it and keep only the part that worked.

The arrow is generated, not drawn. It is priced, and the price is paid only on surprise.
Whether it can also *remember* is the open seam, and this repo is honest that it has not
closed it.

---

## Lineage

Built on the GeometricNeuron / GAIT / PerceptionLab line (`github.com/anttiluode`):
the V13→V14 result that direction must be actively generated, the whorl's topological
latch, and `the_tensor`'s spend-on-surprise economy. The insight and direction are
Antti Luode's; the build, the experiments, and this synthesis were written with Claude
(Opus 4.8). The invitation of the parent line stands here too: **attack the ledger** —
and the place to push is the latch, in §3.

*Do not hype. Do not lie. Just show.*
