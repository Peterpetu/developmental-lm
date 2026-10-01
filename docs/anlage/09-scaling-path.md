# 09 — The scaling path

Non-normative for code; **normative for what may be claimed**.

## 9.1 The rule

**[NORMATIVE]** No component may be described as "scalable" on the strength of the argument
below. The argument establishes only that no component has a *known* barrier. Absence of a
known barrier is not evidence of scaling (SOMA §09.1, retained).

## 9.2 What is scale-free by construction

* `|Gamma|` does not depend on `d_s, L_s, F_s, h_s, V` (§03.6, D4). A lineage regressed at
  350M expresses a 3B primordium by the same decode, with codes for the new units already in
  the tables (`n_c = 13` addresses 8,192 channels; `n_v = 15`, 32,768 tokens). Growing an
  address table costs one feature-map column per bit and is a versioned act.
* The decode costs `2 H d_g` FLOPs per weight and runs a handful of times per lifetime; at
  10⁹ weights it is `≈ 5 × 10^{11}` FLOPs, seconds.
* `REGRESS` is gradient descent on a small network with the weights as data; at 10⁹ targets
  it is minutes and parallelises over roles.

## 9.3 What parallelises

* Siblings are independent processes; only `Gamma` (1.7 MB in float32) and per-stratum
  scalars cross the network. `K` siblings on `K` GPUs cost the wall-clock of one.
* The adult's training is ordinary data-parallel training of a hybrid model; nothing in
  ANLAGE touches the training step (A3), so every production optimisation of the hot path
  (FSDP, kernels, FP8) applies unchanged.
* Inference is a plain 3:1 gated-delta : attention hybrid: linear-time mixers, one attention
  layer in four, a KV cache one quarter of a transformer's.

## 9.4 Continual learning, stated as the extension it is

The token → step consolidation (context into late-block weights, the TTT-E2E / sleep-time
family, §10 E-23) followed by the step → generation consolidation (`REGRESS`) is the same
primitive of §05 at test time. It is a stage-3 experiment; nothing in R1–R8 depends on it and
nothing about it is claimed here.

## 9.5 What production would additionally require

1. Multi-device growth: a growth event reshapes every sharded tensor; the reshaping is
   local per role but the optimiser state's resharding is not designed here.
2. A Pile-scale tokenizer and a document-masked attention path (FlexAttention) for packed
   sequences; the local runs use the standard packed causal stream without document masks.
3. Post-training. Untouched, and carrying no identified incompatibility: the adult is a
   standard hybrid model.
4. Reasoning and agentic evaluation (ARC-class, agentic coding, long-horizon) below `10^9`
   parameters is out of scope (SOMA §08.8, retained); §10 E-28 records why small-model ARC
   results in the literature are dominated by task conditioning and test-time compute rather
   than architecture.

**[NORMATIVE]** Until 1–4 are addressed, ANLAGE is a research architecture with a scaling
*argument*, not a production architecture. Any claim beyond that is unsupported.

## 9.6 Risks

| risk | mitigation | residual |
|---|---|---|
| Windows toolchain: `torch.compile` may degrade to eager; no `flash_attn` module | SDPA path; backend recorded in every manifest; throughput targets measured, not assumed | slower baselines, honestly reported |
| the growth literature's gains shrink above ~20 tokens/parameter | budgets in FLOPs; H-GROW's second inequality re-measured here | growth may not pay at a longer budget; reported |
| merging fails when siblings leave one basin | siblings are short (`f_0/K` of `C_total`); (7.2) guarantees the merge never loses | population reduces to selection only |
| the genome regression underfits high-rank roles | per-role residuals reported; dials raised in a fixed order (§08.4) | larger genome; ratio reported |
| regression to the mean of the adult (hot start with the same trajectory) buys nothing over a warm start | the `eps` dial makes this the first thing measured (R6) | K-LOOP fires; the design reduces to growth + population |
