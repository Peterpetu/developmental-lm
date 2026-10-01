"""What does the model actually write?

SPEC: 09 §9.6 rule 1. Takes the elite of an archive, optionally trains it for
longer than the fitness horizon, and prints samples alongside the loss.
"""
import argparse, dataclasses, json, math, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import torch

# The model emits byte sequences that need not be valid UTF-8, and on Windows a
# cp1252 console cannot print U+FFFD. Replacement characters in a sample are
# themselves a finding, so they are shown rather than suppressed.
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from gnomon.utils.report import write_json
from gnomon.config import GenomeShape, get_config
from gnomon.data.stream import Shard
from gnomon.data.tokenizer import load_tokenizer
from gnomon.eval.generate import describe_loss, generate
from gnomon.genome.arch import ArchHead
from gnomon.genome.decoder import Decoder
from gnomon.lifetime.runner import LifetimeRunner
from gnomon.qd.archive import Archive

ap = argparse.ArgumentParser()
ap.add_argument("--archive", default=None)
ap.add_argument("--config", default="m-pico")
ap.add_argument("--steps", type=int, default=None, help="train this long first")
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--d-z", type=int, default=16)
ap.add_argument("--n-samples", type=int, default=4)
ap.add_argument("--max-new", type=int, default=100)
ap.add_argument("--prompts", nargs="*", default=[
    "Once upon a time", "The little girl", "Tom and Jane went to the",
    "One day, a boy named"])
ap.add_argument("--lr", type=float, default=None)
ap.add_argument("--out", default=None)
a = ap.parse_args()
torch.backends.cuda.matmul.allow_tf32 = True

cfg = get_config(a.config)
if a.steps: cfg = dataclasses.replace(cfg, N_life=a.steps)
cfg = dataclasses.replace(cfg, batch=a.batch,
                          **({"lr": a.lr} if a.lr else {}))
shape = GenomeShape(d_z=a.d_z)
tok = load_tokenizer(f"data/tokenizer_{cfg.vocab}.json")
root = f"data/ts_{cfg.vocab}"
tr, te = Shard.load(root, "train"), Shard.load(root, "test")

if a.archive:
    arch_store = Archive.load(a.archive)
    elite = arch_store.best()
    w = {k[2:]: v for k, v in elite.state.items() if k.startswith("w.")}
    ah = ArchHead(shape)
    ah.load_state_dict({k[2:]: v for k, v in elite.state.items()
                        if k.startswith("a.")})
    arch = ah.decode_arch(cfg)
    print(f"elite of {a.archive}: generation {elite.generation}, "
          f"fitness {elite.fitness:.4f} at the fitness horizon")
else:
    w, arch = Decoder(shape, seed=0).state_dict(), None
    print("no archive given: a *random* genome, as the generation-0 reference")

runner = LifetimeRunner(cfg, shape, train_shard=tr, eval_shard=te,
                        device="cuda", compile_mode=None, n_eval_batches=8)
if arch is not None:
    runner.set_arch(arch)
print(f"architecture [{runner.model.arch.summary()}]  "
      f"{runner.model.n_decoded():,} decoded from {runner.model.n_genome():,} "
      f"genome + {runner.model.n_soma():,} soma reals")
print(f"training {cfg.N_life} steps at batch {cfg.batch} "
      f"({cfg.N_life * cfg.batch * cfg.T / 1e6:.1f}M tokens)\n")

t0 = time.time()
r = runner.run(w, seed=a.seed, progress=False)
print(f"held-out (TEST split) CE:")
print(f"  at birth  {describe_loss(r.birth, cfg.vocab)}")
print(f"  trained   {describe_loss(r.final, cfg.vocab)}")
print(f"  curve     " + "  ".join(f"{s}:{v:.3f}" for s, v in r.curve))
print(f"  {time.time()-t0:.0f}s\n")

W = runner.model.decode()
samples = []
for i, p in enumerate(a.prompts[:a.n_samples]):
    txt = generate(runner.model, tok, prompt=p, max_new=a.max_new,
                   temperature=0.8, top_p=0.95, seed=1000 + i, weights=W)
    samples.append({"prompt": p, "text": txt})
    print(f"[{p!r}]\n  {txt}\n")

if a.out:
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    write_json(a.out, {"archive": a.archive, "config": cfg.name, "N_life": cfg.N_life,
         "birth": r.birth, "final": r.final, "curve": r.curve,
         "arch": runner.model.arch.summary(), "samples": samples})
    print(f"wrote {a.out}")
