# SPORE — the results register

Rendered by `python scripts/spore_analyse.py --markdown` on 2026-09-10 05:36 from `runs/spore/`; every number carries its artifact (`runs/spore/<run>/life.json`, `counts.json`, `evo-*/history.json`). Regenerate after every run; never edit by hand. Verdicts: `docs/spore/02-protocol.md`; the readings and their reasons: `03-evidence-ledger.md`.

```
== lives (test CE per stage; final = the stage the FITNESS set chose)
run                       tokens      F     M   birth     lin    soft   refin   final     fit stage              s
base-5M-s0                 5.01M  16897  1536  8.1204  3.8886  5.8467  3.1222  3.1222  3.1488 refined          308
D2048-05M                  0.51M  15745   384  8.1204  4.3772  5.9138  4.1957  4.1957  4.1920 refined           63
D4096o123-05M              0.51M  21889   384  8.1088  4.3523  5.8825  4.1664  4.1664  4.1678 refined          153
det-05M                    0.51M  15745   384  8.1204  4.3772  5.9138  4.1955  4.1955  4.1919 refined           64
det2a-05M                  0.51M  15745   384  8.1204  4.3772  5.9138  4.1957  4.1957  4.1921 refined           63
det2b-05M                  0.51M  15745   384  8.1204  4.3772  5.9138  4.1954  4.1954  4.1918 refined           65
det3a-05M                  0.51M  15745   384  8.1204  4.3772  5.9138  4.1957  4.1957  4.1920 refined           66
det3b-05M                  0.51M  15745   384  8.1204  4.3772  5.9138  4.1957  4.1957  4.1920 refined           65
E3-5M                      5.01M  16897  1536  8.1204  3.8888  5.8471  2.8361  2.8361  2.8603 refined          575
elite-5M                   5.01M  17665   768  7.6523  3.8218  5.7027  2.8136  2.8136  2.8283 refined          483
final-25M                 25.00M  17921  1536  7.4047  3.6379  5.5784  2.4084  2.4084  2.4297 refined         3211
lr1e-2-5M                  5.01M  16897  1536  8.1204  3.8891  5.8474  2.9758  2.9758  3.0017 refined          309
lr2e-2-5M                  5.01M  16897  1536  8.1204  3.8889  5.8468  2.8901  2.8901  2.9164 refined          308
lr4e-2-5M                  5.01M  16897  1536  8.1204  3.8886  5.8467  2.9048  2.9048  2.9352 refined          308
onehot-5M                  5.01M  25089  1536  8.1204  3.7980  5.8048  3.0840  3.0840  3.1069 refined         2426
s-D1024-5M                 5.01M   9729  1536  7.5905  3.8899  5.7509  3.0604  3.0604  3.0758 refined          180
s-gridge-5M                5.01M  16897  1536  7.5673  3.7467  5.6444  2.9405  2.9405  2.9587 refined          285
s-growoff-5M               5.01M  15361     0  7.5673  3.8118  5.7162  2.9864  2.9864  3.0031 refined          247
s-growrand-5M              5.01M  16897  1536  7.5673  3.8132  5.6594  2.9617  2.9617  2.9795 refined          309
s-kw2048-5M                5.01M  16897  1536  7.5673  3.8117  5.6586  2.9625  2.9625  2.9801 refined          315
s-kw2048-rand-5M           5.01M  16897  1536  7.5673  3.8146  5.6595  2.9617  2.9617  2.9795 refined          315
s-nobind-5M                5.01M  16897  1536  8.0648  3.8205  5.7040  3.0044  3.0044  3.0199 refined          309
s-nomem-5M                 5.01M  15873  1536     -    3.8397  5.9426  3.0893  3.0893  3.1115 refined          251
s-res-5M                   5.01M  18945  1536  7.5673  3.8175  5.6649  2.9313  2.9313  2.9494 refined          388
sem05-05M                  0.51M  15745   384  7.7723  4.2995  5.8239  4.1165  4.1165  4.1054 refined           64
sem05-5M                   5.01M  16897  1536  7.5714  3.8128  5.6608  2.9670  2.9670  2.9805 refined          308
sem05-5M-s1                5.01M  16897  1536  7.5936  3.8099  5.6647  2.9656  2.9656  2.9833 refined          309
sem05-5M-s2                5.01M  16897  1536  7.5633  3.8104  5.6496  2.9667  2.9667  2.9819 refined          309
sem05-d512-5M              5.01M  16897  1536  7.7318  3.8178  5.7056  2.9825  2.9825  2.9999 refined          309
sem05-det-05M              0.51M  15745   384  7.7723  4.2995  5.8239  4.1167  4.1167  4.1055 refined           63
sem05-dk2048-5M            5.01M  17921  1536  7.5172  3.8183  5.6539  2.9438  2.9438  2.9614 refined          376
sem05-E2-5M                5.01M  16897  1536  7.5673  3.8124  5.6595  2.8006  2.8006  2.8179 refined          450
sem05-E3-25M              25.00M  16897  1536  7.4368  3.7041  5.6188  2.4618  2.4618  2.4809 refined         2662
sem05-E3-5M                5.01M  16897  1536  7.5673  3.8117  5.6590  2.7193  2.7193  2.7366 refined          576
sem05-E3-lr1e-2-5M         5.01M  16897  1536  7.5673  3.8121  5.6591  2.6338  2.6338  2.6537 refined          577
sem05-lro-5M               5.01M  16897  1536  7.5594  3.8127  5.6495  2.9644  2.9644  2.9815 refined          309
sem1-5M                    5.01M  16897  1536  7.1702  3.9089  5.6100  2.9961  2.9961  3.0127 refined          309
smoke-05M                  0.51M   8577   384  8.1392  4.4112  6.0111  4.3750  4.3750  4.3755 refined          203
sw-base-1M                 1.02M  16129   768  7.7225  4.0922  5.7692  3.4352  3.4352  3.4438 refined           93
sw-E2-1M                   1.02M  16129   768  7.7225  4.0918  5.7693  3.1487  3.1487  3.1700 refined          119
sw-E2-gb083-1M             1.02M  16129   768  7.7225  4.1068  5.7783  3.1731  3.1731  3.1933 refined          119
sw-E2-res-dk512-1M         1.02M  17665   768  7.8080  4.1038  5.7756  3.1739  3.1739  3.1943 refined          137
sw-gb083-1M                1.02M  16129   768  7.7225  4.1072  5.7778  3.4481  3.4481  3.4559 refined           93
sw-res-1M                  1.02M  18177   768  7.7225  4.0931  5.7637  3.4031  3.4031  3.4121 refined          115

== nulls and the ceiling
counts-05M   unigram 5.9626  bigram 4.1540  trigram 3.8753  (lam2 0.86, lam3 0.47; 0.5M tokens)
counts-5M    unigram 5.9410  bigram 3.8393  trigram 3.3754  (lam2 0.95, lam3 0.66; 5.0M tokens)
counts-25M   unigram 5.9398  bigram 3.7473  trigram 3.1675  (lam2 0.98, lam3 0.78; 25.0M tokens)
B-GD (13M transformer): 0.8M -> 4.306 (train), 1.6M -> 3.205 (train), 2.5M -> 2.855 (train), 10.0M -> 2.2255 (test), 20.0M -> 1.9975 (test), 25.0M -> 1.857 (test, 4 seeds), 201.0M -> 1.804 (test)

G-NOISE  PASS  sigma_e = 0.0007 nats (2 df) over atom seeds; test CE 2.9670, 2.9656, 2.9667; linear-head sigma 0.0015
R-BIRTH  PASS  innate memory-only predictor 7.5714 vs null 9.0109 (delta -1.4395; tau_m 26.08)
R-PARADIGM@5M  sem05-E3-5M test 2.7193  vs trigram 3.3754 (delta -0.6560) -> PASS  | K-SOTA gap to B-GD at 5M (2.536, interpolated): +0.1829
R-PARADIGM@25M  sem05-E3-25M test 2.4618  vs trigram 3.1675 (delta -0.7057) -> PASS  | K-SOTA gap to B-GD at 25M (1.857, interpolated): +0.6048
K-SOTA slope 5M->25M: SPORE -0.1109 nats/doubling vs B-GD -0.2926  -> FIRES (flatter)
K-HDC arms vs sem05-5M (delta > 0 = the ablation hurts):
   s-nobind-5M  final 3.0044 (delta +0.0374; linear-head delta +0.0077) >= 3 sigma
   s-nomem-5M   final 3.0893 (delta +0.1223; linear-head delta +0.0270) >= 3 sigma
   s-D1024-5M   final 3.0604 (delta +0.0934; linear-head delta +0.0771) >= 3 sigma
K-HDC  does not fire  (3 of 3 ablations cost >= 3 sigma_e = 0.0022)
K-DEV  FIRES  select 2.9670 vs off 2.9864 (delta +0.0194) vs random 2.9617 (delta -0.0053); bar 0.0022
K-DEV (11b, ridge-born units)  does not fire  ridge 2.9405 vs off (delta -0.0459) vs random (delta -0.0211); bar 0.0022
K-REFINE  does not fire  refined 2.9670 vs best ridge head 3.8128 (delta -0.8458; bar 0.0022)
G-DET  FAIL  D2048-05M 4.195676 vs det-05M 4.195517 (|delta| 1.59e-04)
G-DET  FAIL  sem05-05M 4.116540 vs sem05-det-05M 4.116691 (|delta| 1.50e-04)
G-DET  FAIL  det2a-05M 4.195728 vs det2b-05M 4.195441 (|delta| 2.87e-04)
G-DET  PASS  det3a-05M 4.195673 vs det3b-05M 4.195653 (|delta| 2.04e-05)

== K-EVO
evo-1M: 32 lives, 4 generations, 1.84 h; base fitness 3.4437 test 3.4352
   gen  0: elite fitness 3.3806  test 3.3452  (pop 3.3806..4.0908)
   gen  1: elite fitness 3.3806  test 3.3452  (pop 3.3806..3.5537)
   gen  2: elite fitness 3.1919  test 3.1727  (pop 3.1919..3.3859)
   gen  3: elite fitness 3.1909  test 3.1703  (pop 3.1909..3.3672)
   elite genome moved: {'s_M': (2, 1627464175), 'd_k': (1024, 512), 'E': (1, 2), 'reservoir': (False, True), 'rho_res': (0.9, 0.6136620644477747)}
   K-EVO (screen): elite - base = -0.2649 on the test set at the screening budget (the pre-registered reading is the elite re-scored at 5M against sem05-5M; see the ledger)
```
