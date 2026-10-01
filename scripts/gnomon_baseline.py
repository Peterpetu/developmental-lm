"""BASE-BIG / BASE-FREE: what does the bottleneck cost?

SPEC: 09 §9.1. The free-weight twin shares `Gnomon.forward` and differs in
exactly one respect -- where the weights come from -- so the comparison is of
the bottleneck and not of two implementations.

  BASE-BIG   matched *decoded* parameter count: the honest upper bound.
  BASE-FREE  matched *free* parameter count: the compression claim.
"""
import argparse, dataclasses, json, math, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import torch
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from gnomon.utils.report import write_json
from gnomon.config import GenomeShape, get_config
from gnomon.data.stream import Shard, TokenStream, eval_batches
from gnomon.data.tokenizer import load_tokenizer
from gnomon.eval.generate import describe_loss, generate
from gnomon.genome.arch import ArchHead
from gnomon.lifetime.train import wsd_lr
from gnomon.pheno.plan import Arch
from gnomon.pheno.twin import FreeTwin
from gnomon.qd.archive import Archive

ap = argparse.ArgumentParser()
ap.add_argument("--archive", default=None, help="match this run's elite architecture")
ap.add_argument("--config", default="m-pico")
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--d-model", type=int, default=None)
ap.add_argument("--d-ff", type=int, default=None)
ap.add_argument("--layers", type=int, default=None)
ap.add_argument("--n-h", type=int, default=None)
ap.add_argument("--d-head", type=int, default=None)
ap.add_argument("--label", default="BASE-BIG")
ap.add_argument("--out", default=None)
a = ap.parse_args()
torch.backends.cuda.matmul.allow_tf32 = True

over = {k: v for k, v in (("d_model", a.d_model), ("d_ff", a.d_ff),
                          ("L", a.layers), ("n_h", a.n_h),
                          ("d_head", a.d_head)) if v is not None}
cfg = dataclasses.replace(get_config(a.config), N_life=a.steps, batch=a.batch,
                          lr=a.lr, **over)
shape = GenomeShape()
if a.archive:
    e = Archive.load(a.archive).best()
    ah = ArchHead(shape)
    ah.load_state_dict({k[2:]: v for k, v in e.state.items() if k.startswith("a.")})
    arch = ah.decode_arch(cfg)
else:
    arch = Arch.uniform(cfg)

twin = FreeTwin(cfg, arch=arch, seed=a.seed).cuda()
tok = load_tokenizer(f"data/tokenizer_{cfg.vocab}.json")
root = f"data/ts_{cfg.vocab}"
tr, te = Shard.load(root, "train"), Shard.load(root, "test")
stream = TokenStream(tr, batch=cfg.batch, T=cfg.T, seed=a.seed ^ 0xA5A5)
ev = eval_batches(te, batch=cfg.batch, T=cfg.T, n_batches=8, device="cuda")

print(f"{a.label}  architecture [{arch.summary()}]  "
      f"{twin.n_params():,} FREE parameters (all of them weights)")
print(f"training {cfg.N_life} steps at batch {cfg.batch} "
      f"({cfg.N_life*cfg.batch*cfg.T/1e6:.1f}M tokens), Muon on matrices + AdamW\n")

opt = torch.optim.AdamW(twin.parameters(), lr=cfg.lr, betas=(0.9, 0.95),
                        weight_decay=0.0)

@torch.no_grad()
def evaluate():
    return sum(float(twin.loss(b)) for b in ev) / len(ev)

t0, curve = time.time(), [(0, evaluate())]
for n in range(cfg.N_life):
    for g in opt.param_groups:
        g["lr"] = wsd_lr(n, cfg)
    loss = twin.loss(stream.next("cuda"))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(twin.parameters(), 1.0)
    opt.step()
    if (n + 1) % max(1, cfg.N_life // 10) == 0:
        curve.append((n + 1, evaluate()))
print(f"held-out (TEST split) CE:")
print(f"  at init   {describe_loss(curve[0][1], cfg.vocab)}")
print(f"  trained   {describe_loss(curve[-1][1], cfg.vocab)}")
print(f"  curve     " + "  ".join(f"{s}:{v:.3f}" for s, v in curve))
print(f"  {time.time()-t0:.0f}s\n")
for i, p in enumerate(["Once upon a time", "The little girl"]):
    print(f"[{p!r}]\n  {generate(twin._host, tok, prompt=p, max_new=60, seed=1000+i, weights=twin.weights())}\n")
if a.out:
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    write_json(a.out, {"arch": arch.summary(), "n_free": twin.n_params(), "curve": curve})
