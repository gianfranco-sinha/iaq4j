"""Merkle tree provenance for the training pipeline.

Six-level tree chaining hashes from sensor leaf to trained model root:

    Sensor (leaf)
      └─→ RawData
            └─→ CleansedData
                  └─→ PreprocessedData
                        └─→ SplitData
                              └─→ TrainedModel (root)

Each node's hash = SHA256(node_type || sorted_content || sorted(child_hashes)).
"""

import hashlib
import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("training.merkle")


# ── Helpers ────────────────────────────────────────────────────────────────


def _json_default(obj):
    """JSON serializer for numpy/non-standard types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _hash_dict(d: dict) -> str:
    """SHA256 of a dict's sorted JSON representation, truncated to 16 hex chars."""
    raw = json.dumps(d, sort_keys=True, default=_json_default)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ── Deterministic hashing utilities ────────────────────────────────────────


def compute_weights_hash(state_dict: dict) -> str:
    """Deterministic SHA256 of model weights (sorted by key)."""
    h = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        h.update(key.encode())
        h.update(state_dict[key].cpu().numpy().tobytes())
    return h.hexdigest()


def compute_scaler_hash(feature_scaler, target_scaler) -> str:
    """Deterministic SHA256 of fitted scaler attributes (no pickle)."""
    attrs = {}

    # StandardScaler fitted attributes
    if hasattr(feature_scaler, "mean_"):
        attrs["feature_mean"] = feature_scaler.mean_.tolist()
    if hasattr(feature_scaler, "var_"):
        attrs["feature_var"] = feature_scaler.var_.tolist()
    if hasattr(feature_scaler, "scale_"):
        attrs["feature_scale"] = feature_scaler.scale_.tolist()

    # MinMaxScaler fitted attributes
    if hasattr(target_scaler, "data_min_"):
        attrs["target_data_min"] = target_scaler.data_min_.tolist()
    if hasattr(target_scaler, "data_max_"):
        attrs["target_data_max"] = target_scaler.data_max_.tolist()
    if hasattr(target_scaler, "data_range_"):
        attrs["target_data_range"] = target_scaler.data_range_.tolist()
    if hasattr(target_scaler, "scale_"):
        attrs["target_scale"] = target_scaler.scale_.tolist()
    if hasattr(target_scaler, "min_"):
        attrs["target_min"] = target_scaler.min_.tolist()

    return hashlib.sha256(
        json.dumps(attrs, sort_keys=True).encode()
    ).hexdigest()


# ── MerkleNode ─────────────────────────────────────────────────────────────


@dataclass
class MerkleNode:
    node_type: str                          # e.g. "sensor", "raw_data", ...
    content_inputs: Dict[str, Any]          # Hashable metadata for this node
    children: List["MerkleNode"] = field(default_factory=list)
    content_hash: str = ""                  # Set by compute_hash()

    def compute_hash(self) -> str:
        """Recursively hash children first, then this node."""
        for child in self.children:
            child.compute_hash()

        h = hashlib.sha256()
        h.update(self.node_type.encode())
        h.update(json.dumps(self.content_inputs, sort_keys=True, default=_json_default).encode())
        for child in sorted(self.children, key=lambda c: c.content_hash):
            h.update(child.content_hash.encode())

        self.content_hash = h.hexdigest()
        return self.content_hash

    def to_dict(self) -> dict:
        """JSON-serializable dict representation."""
        return {
            "node_type": self.node_type,
            "content_inputs": self.content_inputs,
            "content_hash": self.content_hash,
            "children": [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MerkleNode":
        """Reconstruct a MerkleNode tree from a dict."""
        children = [cls.from_dict(c) for c in d.get("children", [])]
        return cls(
            node_type=d["node_type"],
            content_inputs=d["content_inputs"],
            children=children,
            content_hash=d.get("content_hash", ""),
        )


# ── Builder functions (one per level) ──────────────────────────────────────


def build_sensor_node(sensor_id: Optional[str], firmware_version: Optional[str],
                      sensor_type: str) -> MerkleNode:
    return MerkleNode(
        node_type="sensor",
        content_inputs={
            "sensor_id": sensor_id or "unknown",
            "firmware_version": firmware_version or "unknown",
            "sensor_type": sensor_type,
        },
    )


def build_raw_data_node(sensor_node: MerkleNode, source_metadata: dict,
                        raw_row_count: int) -> MerkleNode:
    content = {"raw_row_count": raw_row_count}
    content["source_type"] = source_metadata.get("source_type", "unknown")
    # Copy source-specific keys
    for key in ("measurement", "hours_back", "csv_path", "num_samples", "seed"):
        if key in source_metadata:
            content[key] = source_metadata[key]
    return MerkleNode(
        node_type="raw_data",
        content_inputs=content,
        children=[sensor_node],
    )


def build_cleansed_data_node(raw_data_node: MerkleNode, data_fingerprint: str,
                             clean_row_count: int, rows_dropped: int,
                             issues_summary: str) -> MerkleNode:
    return MerkleNode(
        node_type="cleansed_data",
        content_inputs={
            "data_fingerprint": data_fingerprint,
            "clean_row_count": clean_row_count,
            "rows_dropped": rows_dropped,
            "issues_summary": issues_summary,
        },
        children=[raw_data_node],
    )


def build_preprocessed_data_node(cleansed_data_node: MerkleNode,
                                 schema_fingerprint: str,
                                 feature_stats: dict,
                                 baselines: dict,
                                 scaler_hash: str,
                                 window_size: int,
                                 num_features: int,
                                 sample_count: int) -> MerkleNode:
    return MerkleNode(
        node_type="preprocessed_data",
        content_inputs={
            "schema_fingerprint": schema_fingerprint,
            "feature_stats_hash": _hash_dict(feature_stats) if feature_stats else "",
            "baselines": baselines,
            "scaler_hash": scaler_hash,
            "window_size": window_size,
            "num_features": num_features,
            "sample_count": sample_count,
        },
        children=[cleansed_data_node],
    )


def build_split_data_node(preprocessed_data_node: MerkleNode,
                          test_size: float,
                          train_samples: int,
                          val_samples: int) -> MerkleNode:
    return MerkleNode(
        node_type="split_data",
        content_inputs={
            "test_size": test_size,
            "train_samples": train_samples,
            "val_samples": val_samples,
        },
        children=[preprocessed_data_node],
    )


def build_trained_model_node(split_data_node: MerkleNode,
                             weights_hash: str,
                             metrics: dict,
                             training_config: dict,
                             model_type: str,
                             version: str) -> MerkleNode:
    return MerkleNode(
        node_type="trained_model",
        content_inputs={
            "model_type": model_type,
            "weights_hash": weights_hash,
            "metrics": {k: float(v) for k, v in metrics.items()},
            "epochs": training_config.get("epochs"),
            "learning_rate": training_config.get("learning_rate"),
            "batch_size": training_config.get("batch_size"),
            "version": version,
        },
        children=[split_data_node],
    )


# ── Verification ───────────────────────────────────────────────────────────


def verify_merkle_tree(model_dir) -> dict:
    """Verify a trained model's Merkle tree against on-disk artifacts.

    Returns:
        dict with keys: valid (bool), root_hash, stored_root_hash, mismatches (list)
    """
    import torch

    model_dir = Path(model_dir)
    result = {"valid": True, "root_hash": "", "stored_root_hash": "", "mismatches": []}

    # Load data_manifest.json
    manifest_path = model_dir / "data_manifest.json"
    if not manifest_path.exists():
        result["valid"] = False
        result["mismatches"].append("data_manifest.json not found")
        return result

    with open(manifest_path) as f:
        manifest = json.load(f)

    tree_dict = manifest.get("merkle_tree")
    if tree_dict is None:
        result["valid"] = False
        result["mismatches"].append("No merkle_tree in data_manifest.json")
        return result

    stored_root_hash = manifest.get("merkle_root_hash", "")
    result["stored_root_hash"] = stored_root_hash

    # Reconstruct tree from stored data
    tree = MerkleNode.from_dict(tree_dict)

    # Recompute from stored content_inputs (structural integrity check)
    recomputed_hash = tree.compute_hash()
    result["root_hash"] = recomputed_hash

    if recomputed_hash != stored_root_hash:
        result["valid"] = False
        result["mismatches"].append(
            f"Root hash mismatch: recomputed={recomputed_hash[:16]}..., "
            f"stored={stored_root_hash[:16]}..."
        )

    # Cross-check weights_hash against actual model.pt
    model_path = model_dir / "model.pt"
    if model_path.exists():
        model_data = torch.load(model_path, map_location="cpu")
        state_dict = model_data.get("state_dict", model_data) if isinstance(model_data, dict) else model_data
        actual_weights_hash = compute_weights_hash(state_dict)

        # Walk to trained_model node (root)
        stored_weights_hash = tree_dict.get("content_inputs", {}).get("weights_hash", "")
        if stored_weights_hash and actual_weights_hash != stored_weights_hash:
            result["valid"] = False
            result["mismatches"].append(
                f"trained_model:weights_hash mismatch: "
                f"actual={actual_weights_hash[:16]}..., "
                f"stored={stored_weights_hash[:16]}..."
            )
    else:
        result["valid"] = False
        result["mismatches"].append("model.pt not found")

    # Cross-check scaler_hash against actual scaler files
    feature_scaler_path = model_dir / "feature_scaler.pkl"
    target_scaler_path = model_dir / "target_scaler.pkl"
    if feature_scaler_path.exists() and target_scaler_path.exists():
        with open(feature_scaler_path, "rb") as f:
            feature_scaler = pickle.load(f)
        with open(target_scaler_path, "rb") as f:
            target_scaler = pickle.load(f)
        actual_scaler_hash = compute_scaler_hash(feature_scaler, target_scaler)

        # Find preprocessed_data node's scaler_hash
        stored_scaler_hash = _find_node_content(tree_dict, "preprocessed_data", "scaler_hash")
        if stored_scaler_hash and actual_scaler_hash != stored_scaler_hash:
            result["valid"] = False
            result["mismatches"].append(
                f"preprocessed_data:scaler_hash mismatch: "
                f"actual={actual_scaler_hash[:16]}..., "
                f"stored={stored_scaler_hash[:16]}..."
            )

    return result


def _find_node_content(tree_dict: dict, node_type: str, key: str):
    """Recursively find a content_inputs value in a tree dict."""
    if tree_dict.get("node_type") == node_type:
        return tree_dict.get("content_inputs", {}).get(key)
    for child in tree_dict.get("children", []):
        val = _find_node_content(child, node_type, key)
        if val is not None:
            return val
    return None


# ── Diff ───────────────────────────────────────────────────────────────────


def diff_merkle_trees(old_tree: dict, new_tree: dict) -> dict:
    """Compare two Merkle trees and report which levels changed.

    Args:
        old_tree: tree dict from previous training run
        new_tree: tree dict from current training run

    Returns:
        dict with changed_levels and unchanged_levels lists
    """
    changed = []
    unchanged = []
    _diff_recursive(old_tree, new_tree, changed, unchanged)
    return {"changed_levels": changed, "unchanged_levels": unchanged}


def _diff_recursive(old: dict, new: dict, changed: list, unchanged: list):
    """Recurse through paired nodes, comparing hashes."""
    old_type = old.get("node_type", "")
    new_type = new.get("node_type", "")

    if old_type != new_type:
        changed.append(new_type or old_type)
        return

    old_hash = old.get("content_hash", "")
    new_hash = new.get("content_hash", "")

    if old_hash == new_hash:
        # Subtree unchanged — collect all node types below
        unchanged.append(old_type)
        _collect_types(old, unchanged)
        return

    changed.append(old_type)

    # Recurse into children (paired by index)
    old_children = old.get("children", [])
    new_children = new.get("children", [])
    for oc, nc in zip(old_children, new_children):
        _diff_recursive(oc, nc, changed, unchanged)


def _collect_types(tree_dict: dict, out: list):
    """Collect all descendant node_types into out list."""
    for child in tree_dict.get("children", []):
        out.append(child.get("node_type", ""))
        _collect_types(child, out)
