# 06 — Does the genome carry the function?

The instrument that would have caught SOMA in hour six. §00 measured the answer for SOMA
at **2.4%** after three weeks, 25k lines and two adversarial verification passes, none of
which could have found it: a conformance suite measures conformance to a specification,
and SOMA's specification specified an inert model.

This section makes the question a measurement, gives it a threshold, and puts it on the
smoke run.

## 6.1 Two questions that must not be conflated

**[NORMATIVE]**

* **Q1, architectural.** *Does `Gamma` determine the network's function, or has the soma
  found a path that makes `Gamma` irrelevant?* Answerable **before any evolution**, on the
  smoke run, at generation 0.
* **Q2, evolutionary.** *Does evolving `Gamma` improve it?* This is the generation curve
  of §09 and cannot be answered until §08 has run.

The plan that preceded this specification proposed a single statistic, `share_retrain`,
for both. It cannot serve for Q1: at generation 0 the evolved genome *is* a random genome,
so any statistic phrased as "evolved versus random" is identically zero by construction —
which is precisely the defect (`BirthScore = 0` by construction) that made SOMA's decisive
experiment measure nothing. Q1 needs a statistic that is informative when no evolution has
happened yet.

## 6.2 The primary statistic for Q1: heritability

**[NORMATIVE]** Q1 restated: *do different genomes, given full lifetimes, reach different
outcomes?* If every genome trains to the same loss, then the genome does not matter, the
search has nothing to search, and no amount of selection can produce a response. This is
exactly the SOMA condition.

The quantity is **broad-sense heritability** (Falconer & Mackay, *Introduction to
Quantitative Genetics*, 4th ed., §8; `H^2 = V_G / V_P`), estimated by a fully crossed
design of `n_G` genomes x `n_S` lifetime seeds:

```
    phenotype    P(g, s) = held-out CE after a full lifetime of genome g under seed s
    MSB          = n_S * sum_g (Pbar_g - Pbar)^2 / (n_G - 1)
    MSW          = sum_g sum_s (P(g,s) - Pbar_g)^2 / (n_G (n_S - 1))
    V_G          = max( (MSB - MSW) / n_S , 0 )
    h2           = V_G / (V_G + MSW)
```

with a 95% BCa confidence interval bootstrapped over **genomes** (resampling genomes with
their complete seed sets, so the nesting is preserved) using the vendored
`utils/stats.bca_bootstrap`.

**[NORMATIVE — why this is the right threshold, not an arbitrary one.]** The breeder's
equation `R = h^2 S` states that the response to one generation of selection is the
heritability times the selection differential. Under truncation selection of the top
`p = 1/4` of a population, `S = 1.271 sigma_P`. Therefore

```
    R  =  1.271 * h2 * sigma_P     nats per generation
```

`h2` is not a diagnostic that correlates with evolvability; it is, to first order, the
**constant of proportionality** of evolvability. A floor on it is a floor on the rate at
which §08 can work.

**[NORMATIVE — `KG-HERIT`]** The gate is

```
    h2 >= 0.25        point estimate
    lower bound of the 95% BCa CI  >= 0.10
```

`h2 = 0.25` gives `R = 0.32 sigma_P` per generation, so a 60-generation run is not
budget-limited by heritability. Requiring the interval's lower bound to clear 0.10 makes
the gate a test for *evidence* of heritability rather than a point estimate that a lucky
draw could satisfy.

**[NORMATIVE]** `KG-HERIT` fires on the **smoke run**, before §07's architecture head and
§08's archive are implemented. If it fires, GNOMON's constitution does not hold and the
architecture is redesigned. This ordering is the single most important process decision in
the project and is not negotiable for schedule reasons.

**[NORMATIVE — the prediction this licenses.]** `h2` and `sigma_P` are measured at
generation 0, before the evolutionary run. The breeder's equation then **predicts** the
first generations' response `R`. §09 compares the predicted `R` against the observed
generation-curve slope. A registered quantitative prediction, made before the run and
checkable after it, is what SOMA's protocol never had.

## 6.3 The secondary statistic: the randomise-and-measure probe

**[NORMATIVE]** The cheap, co-adapted probe that produced §00's table, retained because it
is directly comparable to SOMA's number and costs one forward pass:

```
    Delta_Gamma = L(Gamma_rand, Z*) - L*
    Delta_Z     = L(Gamma*, Z_rand) - L*
    share       = Delta_Gamma / (Delta_Gamma + Delta_Z)
```

`Gamma_rand` is drawn from §03.6's initialisation; `Z_rand` from §04.5's. Both are
averaged over 5 draws. SOMA's value, computed this way over its own germline/soma split,
was **0.024**.

**[NORMATIVE]** `share` is **confounded** by co-adaptation: `Z*` was trained against
`Gamma*`, so damaging either partner hurts. It is reported at every checkpoint as a cheap
tripwire — a collapse toward 0 is conclusive evidence of failure — but a healthy value is
**not** evidence of success, and `share` never substitutes for `h2`. Its secondary floor
is `0.5`.

## 6.3.1 `share_retrain`, the statistic §01.6 names

**[NORMATIVE — it was named and never defined, and that is why it was never
computed.]** §01.6 requires `share_retrain` at every certification point and
§01.11 lists its symbol, but this section defined only §6.3's co-adapted probe,
and §09.2's kill-criteria table omitted `KG-SHARE` altogether. So a
[NORMATIVE] statistic with a threshold attached had no definition, no
estimator and no gate.

§6.1 explains why it was set aside and the argument is right as far as it
goes: `share_retrain` cannot answer **Q1**, because at generation 0 the
evolved genome *is* a random genome and the statistic is identically zero by
construction — the same defect (`BirthScore = 0` by construction) that made
SOMA's decisive experiment measure nothing. That disqualifies it as the gate
on the smoke run. It does not disqualify it as the standing report §01.6 asks
for, once evolution has actually run, and both are now true of it.

**[NORMATIVE]** Three losses, all held-out CE in nats on the same split at the
same budget:

| | |
|---|---|
| `L*` | the evolved genome, after a full lifetime |
| `L_randΓ` | a **random** genome, after a full lifetime |
| `L_birth` | the evolved genome with **no lifetime at all** (`Z` at its initialisation, `eps = 0`) |

```
    Delta_Gamma = L_randGamma - L*        what the genome is worth
    Delta_Z     = L_birth     - L*        what the lifetime is worth
    share_retrain = Delta_Gamma / (Delta_Gamma + Delta_Z)
```

**[NORMATIVE — what makes this the *retrained* estimator.]** §6.3's probe
damages one partner and leaves the other co-adapted to the version it was
trained against, so both deltas are inflated by the damage rather than by the
contribution. Here the comparison genome is not damaged: it is *replaced*, and
the soma is retrained from scratch against the replacement. Nothing in the
numerator is a broken model.

**[NORMATIVE — the asymmetry between the two terms is real and is not a defect
in the estimator.]** `Z` is retrained *within* a lifetime; `Gamma` is retrained
only by *evolution*, across generations. "Re-train the other from scratch" can
therefore be executed for the soma and cannot be for the germline inside one
measurement, and the estimator says so rather than pretending to a symmetry it
does not have. What the numerator asks is the question that matters: how much
of the attainable learning survives replacing the genome, **given that the soma
always gets to adapt**. That is exactly the question SOMA answered with 0.024
and never noticed it had.

**[NORMATIVE — `KG-SHARE`]** Fires below **0.50**, as §01.6 states. A negative
delta — the random genome doing better than the evolved one — is reported as-is
and not clamped: at generation 0 it is a coin flip, and later it is a finding.

## 6.4 The at-birth statistic

**[NORMATIVE]** The paper's own headline quantity: held-out CE with the decoded weights
and **no lifetime learning at all** (`Z` at its initialisation, `n = 0`, `eps = 0`).

```
    L_birth(Gamma)      and      birth_share = (L_null - L_birth) / (L_null - L*)
```

`L_null` is the CE of a random genome with untrained soma. At generation 0 `birth_share`
is ~0 and *should* be: a random genome knows nothing. Its trajectory across generations is
the direct analogue of the paper's 94%-at-birth MNIST result and is reported as such.

## 6.5 `d_z` is set by measurement, not by choice

**[NORMATIVE]** `d_z` is the capacity dial: larger `d_z` gives the soma more room, which
buys language quality and simultaneously lets the soma wash out the differences between
genomes — driving `h2`, and therefore evolvability, toward zero. The two effects trade off
monotonically and the trade-off is the project's central empirical object.

**[NORMATIVE — the protocol, fixed before the sweep is run.]** Measure `(L*, h2, share,
birth_share)` at `d_z in {0, 2, 4, 8, 16, 32, 64}` at M-nano, `n_G = 12`, `n_S = 3`. The
operating point for M1 is the **largest** `d_z` whose `h2` lower confidence bound still
clears 0.10. Reporting the whole frontier is required whatever the outcome; selecting the
operating point after seeing it is legitimate precisely because the rule was fixed first.

**[NORMATIVE]** `d_z = 0` is a real configuration and is the purest form of the
architecture: identity is then entirely innate, the soma is empty, and nothing is learned
within a lifetime at all. It is the left endpoint of the frontier and it is expected to be
a poor language model. Reporting it is not optional.

## 6.6 Cost

At M-pico (600-step lifetimes, ~12 s each) the `h2` design of `n_G = 12`, `n_S = 3` is 36
lifetimes, **~7 minutes**. The frontier of §6.5 at M-nano is 7 points x 36 lifetimes x
~40 s, **~6 hours**. The instrument is cheap enough to run at every checkpoint, which is
the entire point of building it before the machinery it guards.

## 6.7 Acceptance tests

**[NORMATIVE]** The instrument is accepted only when all of the following hold.

| # | property | test |
|---|---|---|
| S1 | ANOVA correctness | on synthetic data with known `V_G` and `V_E`, the estimator recovers `h2` to within Monte-Carlo error over 200 replicates |
| S2 | null behaviour | data generated with `V_G = 0` yields `h2` point estimates centred at 0 and CI lower bounds below 0.10 in `>= 95%` of replicates |
| S3 | **positive control** | a planted architecture in which `Gamma` is disconnected from the forward pass yields `h2 ~ 0` and `share ~ 0`, and `KG-HERIT` **fires**. An absence gate that has never fired is not evidence |
| S4 | SOMA reproduction | the §6.3 probe, run against the archived SOMA checkpoint, reproduces `share = 0.024 +/- 0.005` — the instrument is validated against the failure it was built to detect |
| S5 | seed pairing | genome comparisons use identical seed sets; a mutant that unpairs them fails |
| S6 | split hygiene | `h2` is computed on the fitness split, never the test split |
| S7 | determinism | the full `h2` design is reproducible bit-for-bit from its seed |
| S8 | **`share_retrain` is defined and gated** | the estimator of §6.3.1 recovers a known share on constructed inputs, reports a negative delta rather than clamping it, and `KG-SHARE` fires below 0.50 |
