"""Priority ⑥ — machine-readable checkpoint LINEAGE tree with a consistency check.

The chaining/auto-wiring tests each prove one edge of the θ chain in isolation.
Nothing, though, assembles the WHOLE run's checkpoints into one object and asks
"is this lineage internally consistent?" — the question an operator actually has
after a multi-stage run: did every child really descend from the parent it
names, is that parent file still the bytes the child inherited, and is the
architecture stable across every boundary?

This module builds that object from the checkpoints on disk (or explicitly
provided) and answers it by RE-EXECUTING the facts — re-hashing the parent file,
re-reading each child's stored lineage — never by trusting a single "ok" flag.

A node's parent-edge is consistent iff ALL hold:

  * the named parent is present in the tree (no dangling ancestry);
  * the parent FILE's CURRENT sha256 equals the ``parent_checkpoint_sha256`` the
    child stamped when it inherited — i.e. the parent has not changed since
    (this is the check a bare ``parent_checkpoint_id`` string cannot make);
  * the child and parent share a ``model_config_hash`` — θ actually could have
    crossed the boundary (a mismatch means a silent reshape / partial load).

The head of the chain (no parent) is trivially consistent.  ``is_consistent()``
is the AND over every node; ``inconsistencies()`` lists exactly what failed so a
failure is diagnosable, not just a red bit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from goofspiel.training.checkpoint import load_checkpoint, sha256_file


@dataclass
class LineageNode:
    checkpoint_id: str
    path: str
    parent_checkpoint_id: str | None
    parent_checkpoint_sha256: str | None      # parent file's hash AT inherit-time (stored)
    model_config_hash: str | None
    training_stage: str
    optimizer_reset: bool
    self_sha256: str                          # this file's CURRENT hash (recomputed)


@dataclass
class LineageTree:
    nodes: dict[str, LineageNode] = field(default_factory=dict)

    def add(self, node: LineageNode) -> None:
        self.nodes[node.checkpoint_id] = node

    def inconsistencies(self) -> list[dict[str, Any]]:
        """Every broken parent-edge, each as a {node, reason, ...} dict.

        Empty ⇒ the lineage is internally consistent.  Each entry names the node
        and the concrete failure so the caller can act, not merely know it failed.
        """
        problems: list[dict[str, Any]] = []
        for node in self.nodes.values():
            parent_id = node.parent_checkpoint_id
            if parent_id is None:
                continue  # chain head — nothing to inherit from
            parent = self.nodes.get(parent_id)
            if parent is None:
                problems.append({
                    "node": node.checkpoint_id,
                    "reason": "dangling_parent",
                    "parent_checkpoint_id": parent_id,
                })
                continue
            # The parent file must still be the bytes this child inherited from.
            if node.parent_checkpoint_sha256 is None:
                problems.append({
                    "node": node.checkpoint_id,
                    "reason": "missing_parent_sha256",
                    "detail": "child recorded a parent id but no parent content hash",
                })
            elif node.parent_checkpoint_sha256 != parent.self_sha256:
                problems.append({
                    "node": node.checkpoint_id,
                    "reason": "parent_content_changed",
                    "recorded_parent_sha256": node.parent_checkpoint_sha256,
                    "actual_parent_sha256": parent.self_sha256,
                })
            # Architecture must be stable across the boundary or θ could not have
            # transferred cleanly.
            if (
                node.model_config_hash is not None
                and parent.model_config_hash is not None
                and node.model_config_hash != parent.model_config_hash
            ):
                problems.append({
                    "node": node.checkpoint_id,
                    "reason": "config_hash_mismatch_across_edge",
                    "child_hash": node.model_config_hash,
                    "parent_hash": parent.model_config_hash,
                })
        return problems

    def is_consistent(self) -> bool:
        return not self.inconsistencies()

    def chain_order(self) -> list[str]:
        """Checkpoint ids from head to tail, following parent links.

        Assumes the linear θ chain this project uses (each node has ≤1 parent).
        Returns the longest root-to-leaf path; ties are broken by id for
        determinism.  Purely descriptive — not used by the consistency check.
        """
        children: dict[str, list[str]] = {}
        roots: list[str] = []
        for node in self.nodes.values():
            if node.parent_checkpoint_id and node.parent_checkpoint_id in self.nodes:
                children.setdefault(node.parent_checkpoint_id, []).append(node.checkpoint_id)
            else:
                roots.append(node.checkpoint_id)

        best: list[str] = []
        for root in sorted(roots):
            path: list[str] = []
            cur: str | None = root
            while cur is not None:
                path.append(cur)
                kids = sorted(children.get(cur, []))
                cur = kids[0] if kids else None
            if len(path) > len(best):
                best = path
        return best


def build_lineage_tree(checkpoint_paths: list[str | Path]) -> LineageTree:
    """Assemble a :class:`LineageTree` from checkpoint files on disk.

    Each file's metadata supplies the stored lineage; its CURRENT content is
    re-hashed here (``self_sha256``) so a later ``is_consistent()`` compares a
    child's recorded ``parent_checkpoint_sha256`` against the parent's *actual*
    present bytes — catching a parent that changed after the child inherited.
    """
    tree = LineageTree()
    for path in checkpoint_paths:
        p = Path(path)
        if not p.exists():
            continue
        meta = load_checkpoint(p).get("metadata", {})
        tree.add(LineageNode(
            checkpoint_id=str(meta.get("checkpoint_id", p.stem)),
            path=str(p),
            parent_checkpoint_id=meta.get("parent_checkpoint_id"),
            parent_checkpoint_sha256=meta.get("parent_checkpoint_sha256"),
            model_config_hash=meta.get("model_config_hash"),
            training_stage=str(meta.get("training_stage", "")),
            optimizer_reset=bool(meta.get("optimizer_reset", False)),
            self_sha256=sha256_file(p),
        ))
    return tree


def build_lineage_from_run(artifact_dir: str | Path) -> LineageTree:
    """Build the lineage tree for a full-sequence run's artifact directory.

    Discovers the θ-producer checkpoints at their conventional relpaths and
    assembles them.  Missing files are simply absent from the tree (a partial
    run yields a partial-but-consistent tree, not a crash).
    """
    from goofspiel.training.coordinator import _THETA_CHECKPOINT_RELPATH

    art = Path(artifact_dir)
    paths = [art / rel for rel in _THETA_CHECKPOINT_RELPATH.values()]
    return build_lineage_tree(paths)


__all__ = [
    "LineageNode",
    "LineageTree",
    "build_lineage_tree",
    "build_lineage_from_run",
]
