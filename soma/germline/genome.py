"""`Gamma = (Field, Policy, Inventory)` — the germline as one object.

SPEC: 02 §2.2
SPEC: 02 §2.3

Spec v1 listed a fourth component, `Library`, holding gene-side product-key
tables. Those are **deleted**: nothing ever issued a query against them, since
`g_u` is created by `BIRTH` and modified by `MUTATE` as a *set*, never by
retrieval. They were dead parameters consuming ES budget.

Serialisation lives here rather than in a separate module because the content
hash must cover exactly the tensors that constitute the germline, and splitting
the definition from the hash is how the two drift apart.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from soma.config import GERMLINE_SCALE, GermlineScale, PolicyConstants, SomaConfig
from soma.germline.field import Field
from soma.germline.inventory import Inventory
from soma.germline.policy import N_POLICY, Policy
from soma.germline.registry import GermlineRegistry

__all__ = ["Genome", "GenomeMeta", "SCHEMA_VERSION"]

SCHEMA_VERSION = 3
"""Bumped when the on-disk layout or the policy vector's order changes."""


class GermlineHashMismatch(RuntimeError):
    """A germline whose recomputed hash differs from `content_hash` (§02.3)."""


@dataclass
class GenomeMeta:
    """`meta.json` — all fields required (§02.3)."""

    schema_version: int
    lineage_id: str
    G_gen: int
    parent_hash: Optional[str]
    content_hash: str
    config: Dict[str, Any]
    metrics: Dict[str, Any]
    env: Dict[str, Any]
    seeds: Dict[str, int]
    created_utc: str

    def to_json(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "lineage_id": self.lineage_id,
            "G_gen": self.G_gen,
            "parent_hash": self.parent_hash,
            "content_hash": self.content_hash,
            "config": self.config,
            "metrics": self.metrics,
            "env": self.env,
            "seeds": self.seeds,
            "created_utc": self.created_utc,
        }


@dataclass
class Genome:
    """The germline: small, heritable, gradients prohibited.

    SPEC: 02 §2.2
    """

    field_: Field
    policy: Policy
    inventory: Inventory
    scale: GermlineScale = GERMLINE_SCALE
    lineage_id: str = "root"
    G_gen: int = 0
    parent_hash: Optional[str] = None

    # ------------------------------------------------------------------

    @classmethod
    def initialise(
        cls,
        cfg: SomaConfig,
        *,
        generator: Optional[torch.Generator] = None,
        device: torch.device | str = "cpu",
        lineage_id: str = "root",
    ) -> "Genome":
        """Generation-0 germline: random field, configured policy, empty store."""
        f = Field.initialise(cfg.germline, generator=generator, device=device)
        p = Policy.from_constants(
            cfg.policy, pinned=cfg.ablation_pin, device=device
        )
        inv = Inventory(cfg.germline)
        return cls(f, p, inv, cfg.germline, lineage_id, 0, None)

    def register(self, registry: GermlineRegistry) -> None:
        """Put every germline tensor behind the barrier (§02.1.2)."""
        self.field_.register(registry)
        self.policy.register(registry)
        self.inventory.register(registry)

    def constants(self) -> PolicyConstants:
        return self.policy.constants()

    def clone(self) -> "Genome":
        return Genome(
            self.field_.clone(),
            self.policy.clone(),
            self.inventory.clone(),
            self.scale,
            self.lineage_id,
            self.G_gen,
            self.parent_hash,
        )

    def to(self, device: torch.device | str) -> "Genome":
        return Genome(
            self.field_.to(device),
            self.policy.to(device),
            self.inventory.clone(),
            self.scale,
            self.lineage_id,
            self.G_gen,
            self.parent_hash,
        )

    # ------------------------------------------------------------------
    # Size accounting (§09.3)
    # ------------------------------------------------------------------

    def size_reals(self) -> Dict[str, int]:
        """(9.1) broken out by term, so `d|Gamma|/d|U| = 0` is *measurable*.

        SPEC: 09 §9.3

        Gate `G-GERMSIZE` calls this at three population sizes, three widths and
        three depths and asserts the totals are identical. That is a measurement,
        not the assertion §09.3 warns against.
        """
        return {
            "field": int(self.field_.mu.numel() + self.field_.log_sigma.numel()),
            "inventory": self.scale.J_max * self.scale.k_g,
            "policy": N_POLICY,
            "total": int(
                self.field_.mu.numel()
                + self.field_.log_sigma.numel()
                + self.scale.J_max * self.scale.k_g
                + N_POLICY
            ),
        }

    # ------------------------------------------------------------------
    # Serialisation (§02.3)
    # ------------------------------------------------------------------

    def _tensor_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "mu_g": self.field_.mu.detach().cpu().contiguous(),
            "log_sigma_g": self.field_.log_sigma.detach().cpu().contiguous(),
            "policy": self.policy.z.detach().cpu().contiguous(),
            "inventory": self.inventory.packed("cpu").contiguous(),
        }

    def content_hash(self) -> str:
        """BLAKE2b-256 over tensor bytes in **sorted-key order** (§02.3).

        Sorted-key order is what makes the hash reproducible: a dict's insertion
        order is not part of the germline's identity, and two processes that
        built the same germline by different paths must agree.
        """
        h = hashlib.blake2b(digest_size=32)
        tensors = self._tensor_dict()
        for k in sorted(tensors):
            t = tensors[k]
            h.update(k.encode("utf-8"))
            h.update(str(tuple(t.shape)).encode("utf-8"))
            h.update(str(t.dtype).encode("utf-8"))
            h.update(t.numpy().tobytes())
        return h.hexdigest()

    def save(
        self,
        root: str | os.PathLike,
        *,
        metrics: Optional[Mapping[str, Any]] = None,
        seeds: Optional[Mapping[str, int]] = None,
        config_summary: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[Path, Path]:
        """Write `<G_gen>.germline.safetensors` and `<G_gen>.meta.json`."""
        from safetensors.torch import save_file

        d = Path(root) / "genomes" / self.lineage_id
        d.mkdir(parents=True, exist_ok=True)
        tensors = self._tensor_dict()
        ch = self.content_hash()

        weights_path = d / f"{self.G_gen}.germline.safetensors"
        meta_path = d / f"{self.G_gen}.meta.json"
        save_file(tensors, str(weights_path))

        import datetime

        meta = GenomeMeta(
            schema_version=SCHEMA_VERSION,
            lineage_id=self.lineage_id,
            G_gen=self.G_gen,
            parent_hash=self.parent_hash,
            content_hash=ch,
            config=dict(config_summary or {}),
            metrics=dict(metrics or {}),
            env=environment_block(),
            seeds=dict(seeds or {}),
            created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )
        # The Inventory carries per-archetype statistics that do not fit the
        # packed tensor; they travel in meta.json and are re-attached at load.
        payload = meta.to_json()
        payload["inventory_stats"] = self.inventory.to_json()
        # **[NORMATIVE — the pin set is part of the artifact; audit A-89]** A
        # pinned constant has no preimage under its link, so `Policy` stores
        # `0.0` in its slot and the pin supplies the value on every read. Save
        # the pins beside the weights or the arm's identity is lost: a germline
        # inherited from an ablation run would decode `0.0` through the link and
        # silently become the main arm.
        payload["ablation_pin"] = dict(self.policy.pinned)
        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return weights_path, meta_path

    @classmethod
    def load(
        cls,
        root: str | os.PathLike,
        lineage_id: str,
        G_gen: int,
        *,
        scale: GermlineScale = GERMLINE_SCALE,
        pinned: Optional[Mapping[str, Any]] = None,
        device: torch.device | str = "cpu",
    ) -> Tuple["Genome", GenomeMeta]:
        """Load and **refuse on hash mismatch** (§02.3).

        **[NORMATIVE — the pin set travels with the germline; v3.1, audit A-89]**
        `Policy.from_constants` deliberately writes `0.0` into a pinned slot —
        "a value the pin overrides on every read" — because a pinned constant
        has no preimage under its link. `load` then decoded that `0.0` **through
        the link** as if it were a real value, and no caller in the repository
        passed `pinned=`. Measured on `m0_config(M_NANO)`, saving and loading:

        ```
          name          in-memory   after load
          f_oe                  0        0.25
          k_tok                 1       0.775
          lambda_wire           0    0.693147
          rho_0                 0         0.1
        ```

        No exception: §02.3's hash guard passes, because pins are **not hashed**
        — the germline's content is the same file either way. So
        `--ablation lambda_wire=0 --genome ...` ran the **main arm's constants
        under the ablation arm's label**, and §02.6.2's M-nano -> M1 inheritance
        lost every pin.

        The pin set is now written into `meta.json` beside the weights and read
        back here, with an explicit `pinned=` argument taking precedence so a
        caller can still re-pin deliberately. A pin present in the file and
        **contradicted** by the caller is an error, not a silent override: those
        are two different experiments.
        """
        from safetensors.torch import load_file

        d = Path(root) / "genomes" / lineage_id
        weights_path = d / f"{G_gen}.germline.safetensors"
        meta_path = d / f"{G_gen}.meta.json"
        if not weights_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"no germline at {weights_path}")

        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
            raise GermlineHashMismatch(
                f"germline schema version {payload.get('schema_version')!r} != "
                f"{SCHEMA_VERSION}; the policy vector's order is part of the "
                f"format and a mismatched file cannot be decoded safely"
            )

        tensors = load_file(str(weights_path))
        f = Field(
            tensors["mu_g"].to(device), tensors["log_sigma_g"].to(device), scale
        )
        stored_pins = dict(payload.get("ablation_pin") or {})
        if pinned is None:
            merged = stored_pins
        else:
            conflicts = {
                k: (stored_pins[k], pinned[k])
                for k in set(stored_pins) & set(pinned)
                if stored_pins[k] != pinned[k]
            }
            if conflicts:
                raise GermlineHashMismatch(
                    f"ablation pins in the germline disagree with the caller's: "
                    f"{conflicts}. A pinned constant identifies the arm an "
                    f"experiment belongs to (§02.2.3); silently taking one over "
                    f"the other would run one arm under the other's label."
                )
            merged = {**stored_pins, **pinned}
        p = Policy(tensors["policy"].to(device), merged)
        inv = Inventory.from_json(payload.get("inventory_stats", []), scale)
        genome = cls(
            f, p, inv, scale, lineage_id, G_gen, payload.get("parent_hash")
        )

        recomputed = genome.content_hash()
        if recomputed != payload["content_hash"]:
            raise GermlineHashMismatch(
                f"germline content hash mismatch at {weights_path}:\n"
                f"  recorded   {payload['content_hash']}\n"
                f"  recomputed {recomputed}\n"
                f"A germline whose recomputed hash differs is refused at load "
                f"(§02.3)."
            )

        meta = GenomeMeta(
            schema_version=payload["schema_version"],
            lineage_id=payload["lineage_id"],
            G_gen=payload["G_gen"],
            parent_hash=payload.get("parent_hash"),
            content_hash=payload["content_hash"],
            config=payload.get("config", {}),
            metrics=payload.get("metrics", {}),
            env=payload.get("env", {}),
            seeds=payload.get("seeds", {}),
            created_utc=payload.get("created_utc", ""),
        )
        return genome, meta


def environment_block() -> Dict[str, Any]:
    """`env` for `meta.json` and the run manifest (§02.3, §07.7)."""
    env: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        env["gpu"] = props.name
        env["gpu_memory_gib"] = round(props.total_memory / 2**30, 3)
        env["sm"] = f"{props.major}.{props.minor}"
        try:
            env["driver"] = torch._C._cuda_getDriverVersion()  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - driver query is best-effort
            env["driver"] = None
    else:
        env["gpu"] = None
    return env
