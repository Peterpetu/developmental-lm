"""M1: the language run.

SPEC: 09 §9.1, SPEC: 09 §9.6

The headline configuration -- `512 x 12`, vocab 8192, `T = 512`, `N_life =
20000` -- decoded from an elite genome evolved at M-pico, which is what makes it
§02.7's transfer claim in its strongest form: the genome has never seen this
scale.

**[NORMATIVE — what is reported, and one thing that is not.]** §09.1's row reads
"held-out CE, samples, behavioural suite". The first two are here, on the
**test** split, with bits-per-byte alongside as §09.6 rule 1 requires. The third
is **not**, and is listed rather than silently dropped (§09.6 rule 4):

    SOMA's behavioural regression suite (`soma/eval/regression.py`) grades
    PARITY, MODCOUNT, PERM, DYCK, MQAR and COPY, serialised over a reserved
    symbol block. It is a suite of *formal-language tasks a model was trained
    on*. GNOMON trains on TinyStories and nothing else (§05.2: "no token
    selection, no auxiliary loss, no metabolic term"), so it has never seen
    those tasks and its accuracy on them would be chance by construction. The
    suite is inapplicable to this configuration, not merely unrun, and §00.4's
    "verified and reused: the behavioural regression suite" overstates what
    carried across for that reason. What replaces it here is what a language
    model can actually be held to: held-out CE, bits-per-byte, and samples in a
    fixed decoding.

The run is checkpointed, because eight hours without one means a crash at hour
seven costs eight (§05.7 L8).

    python scripts/gnomon_m1.py --archive runs/gnomon/evo-GNOMON-s0 --steps 20000
    python scripts/gnomon_m1.py --resume                       # after a crash
"""
import argparse, dataclasses, json, math, pathlib, sys, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from gnomon.utils.report import write_json
from gnomon.config import GenomeShape, get_config
from gnomon.data.stream import Shard
from gnomon.data.tokenizer import load_tokenizer
from gnomon.eval.generate import bits_per_byte, describe_loss, generate
from gnomon.genome.arch import ArchHead
from gnomon.genome.decoder import Decoder
from gnomon.lifetime.runner import LifetimeRunner
from gnomon.qd.archive import Archive

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="m1")
ap.add_argument("--archive", default=None,
                help="an evolved archive; its elite genome and architecture are "
                     "decoded at this scale. Omitted = a random genome, which "
                     "is the generation-0 reference KG-SCALE compares against")
ap.add_argument("--steps", type=int, default=None)
ap.add_argument("--batch", type=int, default=None)
ap.add_argument("--lr", type=float, default=None)
ap.add_argument("--d-z", type=int, default=16)
# SPEC: 04 §4.4 -- the three configurations are "fixed now so that no
# configuration is chosen after seeing a result". Any override below therefore
# takes the run OUT of §04.4's set, and the manifest records that in as many
# words so no number from it can be quoted as a §09.1 result.
ap.add_argument("--T", type=int, default=None)
ap.add_argument("--window", type=int, default=None)
ap.add_argument("--d-model", type=int, default=None)
ap.add_argument("--layers", type=int, default=None)
ap.add_argument("--d-ff", type=int, default=None)
ap.add_argument("--n-eval-batches", type=int, default=16)
# A long run needs a curve fine enough to see a stall. The default of
# `N_life // 10` gives ten points over 13 hours, which is too coarse to
# notice that a run has plateaued at the unigram entropy.
ap.add_argument("--eval-every", type=int, default=None)
# SPEC: 05 §5.4 -- §5.4 specifies decoupled weight decay 0 and R-ADAM pins
# the rule's C term to 0, so nothing bounds ||Z||. The 40,000-step run
# diverged: z -> inf, decoder responsiveness 0.321 -> 0.001, decoded
# erank 0.2507 -> 0.0065, and held-out CE slid back to the unigram
# entropy. This enters as a floor under the decoded C.
ap.add_argument("--weight-decay", type=float, default=0.0)
ap.add_argument("--max-identity-rms", type=float, default=None)
ap.add_argument("--no-probe", action="store_true",
                help="turn off the erank/saturation probe (SPEC: 03 §3.8)")
ap.add_argument("--lr-source", default=None,
                help="artifact and horizon the --lr was measured at "
                     "(SPEC: 10 §10.6); recorded in the manifest")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--token-code", default="freq", choices=["freq", "bijection"])
ap.add_argument("--compile-mode", default=None,
                choices=[None, "cudagraph", "reduce-overhead", "default"])
ap.add_argument("--checkpoint-every", type=int, default=500)
ap.add_argument("--resume", action="store_true")
ap.add_argument("--n-samples", type=int, default=6)
ap.add_argument("--max-new", type=int, default=140)
ap.add_argument("--prompts", nargs="*", default=[
    "Once upon a time", "The little girl", "Tom and Jane went to the",
    "One day, a boy named", "There was a big", "Lily saw a"])
ap.add_argument("--out", default="runs/gnomon/m1")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = ap.parse_args()

torch.backends.cuda.matmul.allow_tf32 = True
over = {k: v for k, v in (("N_life", a.steps), ("batch", a.batch),
                         ("lr", a.lr), ("T", a.T), ("window", a.window),
                         ("d_model", a.d_model), ("L", a.layers),
                         ("d_ff", a.d_ff)) if v is not None}
base_cfg = get_config(a.config)
# SPEC: 10 §10.6 -- a rate given on the command line has a provenance too, and
# it is not the configuration's. Carrying the config's string here made a run
# that HAD been measured announce itself as unmeasured, which is the same class
# of error in the other direction.
if a.lr is not None:
    over["lr_provenance"] = (
        f"SET ON THE COMMAND LINE to {a.lr:g}. See --lr-source for what it was "
        f"measured against; unset means it was not measured at all.")
if a.lr is not None and a.lr_source:
    over["lr_provenance"] = (
        f"measured: {a.lr_source}; this run is at N_life="
        f"{a.steps if a.steps else base_cfg.N_life}, so the rate is an "
        f"extrapolation IN HORIZON from where it was measured (§10.6)")
# SPEC: 04 §4.4. `N_life`, `batch` and `lr` are run parameters; the rest are the
# body plan, and changing one means this is no longer that configuration.
SHAPE_FIELDS = ("T", "window", "d_model", "L", "d_ff")
shape_over = {k: v for k, v in over.items() if k in SHAPE_FIELDS}
normative = not shape_over
cfg = dataclasses.replace(base_cfg, **over)
if not normative:
    cfg = dataclasses.replace(cfg, name=f"{base_cfg.name}-custom")
shape = GenomeShape(d_z=a.d_z)
out = pathlib.Path(a.out)
out.mkdir(parents=True, exist_ok=True)
root = f"data/ts_{cfg.vocab}"
tr, te = Shard.load(root, "train"), Shard.load(root, "test")
tok = load_tokenizer(f"data/tokenizer_{cfg.vocab}.json")

arch = None
if a.archive:
    store = Archive.load(pathlib.Path(a.archive) / "archive.pt"
                         if pathlib.Path(a.archive).is_dir() else a.archive)
    elite = store.best()
    w = {k[2:]: v for k, v in elite.state.items() if k.startswith("w.")}
    ah = ArchHead(shape)
    ah.load_state_dict({k[2:]: v for k, v in elite.state.items()
                        if k.startswith("a.")})
    # SPEC: 02 §2.7 -- decoded at *this* configuration, not the one it evolved
    # at. That is the transfer, and it is zero-shot: `|Gamma|` does not change.
    arch = ah.decode_arch(cfg)
    print(f"elite of {a.archive}: generation {elite.generation}, fitness "
          f"{elite.fitness:.4f} at the fitness horizon"
          f"{'  [A-LAMARCK]' if elite.lamarck else ''}")
else:
    w = Decoder(shape, seed=0).state_dict()
    print("no archive given: a RANDOM genome -- the generation-0 reference")

# SPEC: 09 §9.6 rule 1 -- every reported number is on the test split, which
# selection never saw.
runner = LifetimeRunner(cfg, shape, train_shard=tr, eval_shard=te,
                        device=a.device, compile_mode=a.compile_mode,
                        n_eval_batches=a.n_eval_batches,
                        token_code=a.token_code)
if arch is not None:
    runner.set_arch(arch)

if not normative:
    print("*** NOT one of §04.4's three configurations. Overridden body-plan "
          f"fields: {sorted(shape_over)}.")
    print("    §04.4 fixes its configurations 'so that no configuration is "
          "chosen after seeing a result',")
    print("    and these were chosen after a throughput benchmark. No number "
          "from this run is a §09.1 result.")
print(f"{cfg.name}  [{runner.model.arch.summary()}]")
print(f"  {runner.model.n_decoded():,} decoded from {runner.model.n_genome():,} "
      f"genome + {runner.model.n_soma():,} soma reals "
      f"({runner.model.n_decoded() / runner.model.n_genome():,.0f}x)")
print(f"  {cfg.N_life:,} steps x batch {cfg.batch} x T {cfg.T} = "
      f"{cfg.N_life * cfg.batch * cfg.T / 1e6:.0f}M tokens  lr {cfg.lr}")
# SPEC: 10 §10.6 -- a rate is only meaningful with the horizon it was measured
# at, so the run says which it has. An unmeasured one is loud about it: that is
# the whole point of recording provenance rather than only the number.
print(f"  lr provenance: {cfg.lr_provenance}")
if cfg.lr_provenance.startswith("NOT MEASURED"):
    print(f"  *** this run uses an UNMEASURED learning rate. §10.6 records "
          f"using one as a defect in every comparison it touches. Pass --lr "
          f"with a value from scripts/gnomon_lrsweep.py, or accept that the "
          f"result is reported as being at an untuned rate.")
print(f"  backend {runner.backend}  token code {runner.token_code}  "
      f"checkpoint every {a.checkpoint_every} -> {out / 'life.pt'}")
print(f"  ln V = {math.log(cfg.vocab):.4f}\n", flush=True)

t0 = time.time()
r = runner.run(w, seed=a.seed, progress=True, eval_every=a.eval_every,
               weight_decay=a.weight_decay,
               max_identity_rms=a.max_identity_rms,
               probe=not a.no_probe,
               checkpoint=str(out / "life.pt"),
               checkpoint_every=a.checkpoint_every, resume=a.resume)

# SPEC: 09 §9.6 rule 1 -- nats, and bits-per-byte alongside for comparability.
# `tokens_per_byte` is measured from this corpus rather than assumed: the ratio
# is what makes BPB comparable across tokenizers, so guessing it would defeat
# the point of reporting it.
n_bytes = sum(len(tok.decode([int(t)]).encode("utf-8"))
              for t in te.tokens[:200_000])
tpb = 200_000 / max(n_bytes, 1)
print(f"\nheld-out (TEST split) CE, {cfg.N_life:,} steps, {time.time()-t0:.0f}s:")
print(f"  at birth  {describe_loss(r.birth, cfg.vocab, tokens_per_byte=tpb)}")
print(f"  trained   {describe_loss(r.final, cfg.vocab, tokens_per_byte=tpb)}")
print(f"  bits per byte: {bits_per_byte(r.final, 200_000, n_bytes):.4f}"
      f"   ({tpb:.4f} tokens/byte, measured on the test split)")
print(f"  learned {r.birth - r.final:+.4f} nats; late slope {r.slope:.4f}/decade")

W = runner.model.inference_weights()
samples = []
for i, prompt in enumerate(a.prompts[:a.n_samples]):
    txt = generate(runner.model, tok, prompt=prompt, max_new=a.max_new,
                   temperature=0.8, top_p=0.95, seed=1000 + i, weights=W,
                   device=a.device)
    samples.append({"prompt": prompt, "text": txt})
    print(f"\n[{prompt!r}]\n  {txt}")

write_json(out / "m1.json", {
    "config": cfg.name, "base_config": base_cfg.name,
    "normative_config": normative, "overrides": {k: v for k, v in over.items()},
    "full_config": dataclasses.asdict(cfg),
    "N_life": cfg.N_life, "batch": cfg.batch, "T": cfg.T,
    "lr": cfg.lr, "lr_provenance": cfg.lr_provenance, "d_z": a.d_z,
    "weight_decay": a.weight_decay,
    "max_identity_rms": a.max_identity_rms,
    "health": r.health,
    "seed": a.seed,
    "archive": a.archive, "arch": runner.model.arch.summary(),
    "token_code": runner.token_code, "provenance": runner.token_provenance,
    "backend": runner.backend, "lamarck": r.lamarck,
    "n_genome": runner.model.n_genome(), "n_soma": runner.model.n_soma(),
    "n_decoded": runner.model.n_decoded(),
    "birth": r.birth, "final": r.final, "curve": r.curve,
    "bits_per_byte": bits_per_byte(r.final, 200_000, n_bytes),
    "tokens_per_byte": tpb, "seconds": r.seconds,
    "samples": samples,
    # SPEC: 09 §9.6 rule 4 -- an arm that did not run is named, with the reason.
    "not_run": {
        "behavioural_suite": "inapplicable: it grades formal-language tasks a "
                             "model was trained on, and GNOMON trains only on "
                             "TinyStories (§05.2), so its accuracy would be "
                             "chance by construction",
    },
})
print(f"\nwrote {out / 'm1.json'}")
