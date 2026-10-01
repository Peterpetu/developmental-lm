# TEMPO — the register

Rendered by `scripts/tempo_register.py` on 2026-09-04 19:37 from `runs/tempo/`; every number carries its artifact. SPEC: 10 §10.4. Regenerate after every run; never edit by hand.

## Verdicts (10 §10.2, §10.5)

```
R0 SPAN  ceiling 0.00137 nats/token (I1 ok); noise gap 0.00013 (I2 ok)
R1 NOISE  n=1 test 2.4067 sigma_e nan tok/s 100369 mem 5.72 GB
K-CONT                     NOT EVALUATED (an arm is missing or incomplete)
R3 STAT/CRIT  NOT EVALUATED
K-GENE vs a_gamma0         NOT EVALUATED (an arm is missing or incomplete)
K-GENE vs b_homeo          NOT EVALUATED (an arm is missing or incomplete)
K-GENE vs c_random         NOT EVALUATED (an arm is missing or incomplete)
K-GENE vs d_free           NOT EVALUATED (an arm is missing or incomplete)
K-GENE vs e_svf            NOT EVALUATED (an arm is missing or incomplete)
K-GENE vs f_es             NOT EVALUATED (an arm is missing or incomplete)
K-GENE vs g_grpo           NOT EVALUATED (an arm is missing or incomplete)
K-GENE -> NOT EVALUATED
K-HORIZON                  NOT EVALUATED (R4 tempo or (a) missing)
K-REV                      NOT EVALUATED (an arm is missing or incomplete)
K-SAL                      NOT EVALUATED (an arm is missing or incomplete)
K-MORPH vs schedule        NOT EVALUATED (an arm is missing or incomplete)
K-MORPH vs oneshot         NOT EVALUATED (an arm is missing or incomplete)
K-MORPH vs flat            NOT EVALUATED (an arm is missing or incomplete)
K-MORPH -> NOT EVALUATED
K-CRIT  NOT EVALUATED (R3 not run)
K-TRANSFER                 NOT EVALUATED (an arm is missing or incomplete)
K-XFER control             NOT EVALUATED (an arm is missing or incomplete)
K-XFER vision              NOT EVALUATED (an arm is missing or incomplete)
K-XFER speech              NOT EVALUATED (an arm is missing or incomplete)
K-XFER pattern             NOT EVALUATED (an arm is missing or incomplete)
K-XFER -> NOT EVALUATED (0 of 0 evaluated, 4 required)
K-BLEND  NOT EVALUATED
G-ACT                      NOT EVALUATED (no G-ACT run)
K-SOTA                     NOT EVALUATED (an arm is missing or incomplete)
```

## Runs

| run | seed | budget | score | second | per-block Λ range | throughput, memory | artifact |
|---|---|---|---|---|---|---|---|
| *(every life row is a complete run; unfinished lives are not listed)* | | | | | | | | |
| `codec-ts-s0` | 0 | 199,999,488 tokens | ceiling 0.00137 | noise gap 0.00013 (I1 True, I2 True) | — | 694 s | `runs/tempo/codec-ts-s0/result.json` |
| `codec-tsk2-s0` | 0 | 199,999,488 tokens | ceiling 0.00071 | noise gap 0.00003 (I1 True, I2 True) | — | 698 s | `runs/tempo/codec-tsk2-s0/result.json` |
| `R1-nolat-ts-s0` | 0 | 1 ep × N=1525 B=32, F 6.795e+14 | test 2.4067 | reward +nan | Λ/K [+nan, +nan] | 100369 tok/s, 5.72 GB | `runs/tempo/R1-nolat-ts-s0/record.json` |

## Numbers

```
{
 "R0": {
  "ceiling_nll_per_token": 0.0013744140318197797,
  "noise_gap_per_token": 0.00013001659675681776,
  "I1": true,
  "I2": true,
  "steps": 12207,
  "batch_spans": 4096,
  "tokens": 199999488,
  "n_params": 9650176,
  "seconds": 693.5334422588348,
  "n_eval_spans": 33024,
  "complete": true,
  "complete_backfilled": "the run predates the `complete` flag (11 \u00a7P-C.37); its log and codec.pt show it finished, and the flag was added to the record rather than silently assumed by the reader"
 },
 "R1": {
  "tests": [
   2.4067394733428955
  ],
  "sigma_e": null,
  "mean": 2.4067394733428955,
  "tokens_per_s": [
   100369.09411138826
  ],
  "peak_mem_gb": [
   5.720059904
  ],
  "gstat": [
   true
  ],
  "Lambda": [
   -0.511059045791626
  ]
 }
}
```
