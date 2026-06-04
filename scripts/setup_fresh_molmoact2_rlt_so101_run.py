#!/usr/bin/env python3
"""Prepare a fresh versioned MolmoAct2 SO101 online RLT run.

The script copies the reusable first-47 demo replay into a new versioned demo
buffer, creates an empty online-only replay buffer, and updates the active DRTC
experiment YAML to point at the new files.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lerobot.rl.rlt_buffer import RLTReplayBuffer  # noqa: E402


DEFAULT_CONFIG = ROOT / "examples/experiments/configs/baseline_molmoact2_rlt_so101_online.yaml"
DEFAULT_SOURCE_DEMO = (
    ROOT / "outputs/rlt_molmoact2_so101_online/rlt_online_replay_20260831_v2_first47_rollouts.pt"
)
DEFAULT_BUFFER_DIR = ROOT / "outputs/rlt_molmoact2_so101_online"
DEFAULT_OUTPUT_DIR_TEMPLATE = "outputs/rlt_molmoact2_so101_online_20260602_first47_residual_paperish_v{version}"
DEFAULT_DEMO_TEMPLATE = (
    "outputs/rlt_molmoact2_so101_online/"
    "rlt_demo_replay_20260831_v3_first47_residual_paperish_v{version}.pt"
)
DEFAULT_ONLINE_TEMPLATE = (
    "outputs/rlt_molmoact2_so101_online/"
    "rlt_online_replay_20260831_v3_online_residual_paperish_v{version}.pt"
)
DEFAULT_ARCHIVE_TEMPLATE = (
    "outputs/rlt_molmoact2_so101_online/"
    "rlt_online_replay_20260831_v3_online_residual_paperish_v{version}_archive.pt"
)
DEFAULT_WANDB_TEMPLATE = "baseline_molmoact2_rlt_so101_online_residual_paperish_v{version}"
VERSION_RE = re.compile(r"residual_paperish_v(\d+)")


def _repo_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return ROOT / path


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}, got {type(data).__name__}")
    return data


def _versions_from_text(value: Any) -> set[int]:
    if value is None:
        return set()
    return {int(match.group(1)) for match in VERSION_RE.finditer(str(value))}


def _next_version(config: dict[str, Any], buffer_dir: Path) -> int:
    versions: set[int] = set()
    for key in (
        "rlt_output_dir",
        "rlt_demo_buffer_path",
        "rlt_online_buffer_path",
        "rlt_review_archive_path",
        "rlt_wandb_run_name",
    ):
        versions.update(_versions_from_text(config.get(key)))

    if buffer_dir.exists():
        for path in buffer_dir.glob("*residual_paperish_v*.pt"):
            versions.update(_versions_from_text(path.name))

    outputs_dir = ROOT / "outputs"
    if outputs_dir.exists():
        for path in outputs_dir.glob("*residual_paperish_v*"):
            versions.update(_versions_from_text(path.name))

    return (max(versions) if versions else 0) + 1


def _replace_config_keys(path: Path, updates: dict[str, str]) -> None:
    original = path.read_text(encoding="utf-8")
    lines = original.splitlines()
    found: set[str] = set()
    patterns = {
        key: re.compile(rf"^({re.escape(key)}):(?:\s.*)?$")
        for key in updates
    }

    for idx, line in enumerate(lines):
        for key, pattern in patterns.items():
            if pattern.match(line):
                lines[idx] = f"{key}: {updates[key]}"
                found.add(key)
                break

    missing = [key for key in updates if key not in found]
    if missing:
        lines.append("")
        lines.append("# Fresh RLT run paths.")
        for key in missing:
            lines.append(f"{key}: {updates[key]}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _inspect_demo(path: Path) -> tuple[int, int]:
    replay = RLTReplayBuffer.load(path, apply_review_sidecar=True)
    episode_ids = {
        int(sample.episode_id)
        for sample in replay.samples()
        if sample.episode_id is not None
    }
    return len(replay), len(episode_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a fresh versioned MolmoAct2 SO101 online RLT run by copying "
            "the reusable first-47 demo buffer and updating the baseline YAML."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Experiment YAML to update.")
    parser.add_argument(
        "--source-demo-buffer",
        type=Path,
        default=DEFAULT_SOURCE_DEMO,
        help="Reusable 47-demo replay buffer to copy.",
    )
    parser.add_argument(
        "--buffer-dir",
        type=Path,
        default=DEFAULT_BUFFER_DIR,
        help="Directory for versioned demo/online replay buffers.",
    )
    parser.add_argument("--version", type=int, default=None, help="Version to create. Default: next vN.")
    parser.add_argument("--force", action="store_true", help="Overwrite destination buffers if they exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned changes without writing.")
    parser.add_argument(
        "--allow-non-47-demo",
        action="store_true",
        help="Do not fail if the source demo buffer does not contain exactly 47 episodes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = _repo_path(args.config)
    source_demo = _repo_path(args.source_demo_buffer)
    buffer_dir = _repo_path(args.buffer_dir)

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not source_demo.exists():
        raise FileNotFoundError(f"Source demo buffer not found: {source_demo}")

    config = _load_yaml(config_path)
    sample_count, episode_count = _inspect_demo(source_demo)
    if episode_count != 47 and not args.allow_non_47_demo:
        raise ValueError(
            f"Expected source demo buffer to contain 47 episodes, got {episode_count} "
            f"episodes and {sample_count} samples: {source_demo}"
        )

    version = args.version if args.version is not None else _next_version(config, buffer_dir)
    if version <= 0:
        raise ValueError(f"Version must be positive, got {version}")

    new_output_dir = DEFAULT_OUTPUT_DIR_TEMPLATE.format(version=version)
    new_demo_path = DEFAULT_DEMO_TEMPLATE.format(version=version)
    new_online_path = DEFAULT_ONLINE_TEMPLATE.format(version=version)
    new_archive_path = DEFAULT_ARCHIVE_TEMPLATE.format(version=version)
    new_wandb_name = DEFAULT_WANDB_TEMPLATE.format(version=version)

    demo_dest = _repo_path(new_demo_path)
    online_dest = _repo_path(new_online_path)
    output_dest = _repo_path(new_output_dir)

    for dest in (demo_dest, online_dest):
        if dest.exists() and not args.force:
            raise FileExistsError(f"Destination already exists, pass --force to overwrite: {dest}")

    updates = {
        "rlt_output_dir": new_output_dir,
        "rlt_demo_buffer_path": new_demo_path,
        "rlt_online_buffer_path": new_online_path,
        "rlt_wandb_run_name": new_wandb_name,
        "rlt_review_archive_path": new_archive_path,
    }
    training_command = f"./scripts/run_drtc_experiment.sh --viz --config {_repo_rel(config_path)}"

    print(f"Preparing MolmoAct2 SO101 RLT fresh run v{version}")
    print(f"  config:       {_repo_rel(config_path)}")
    print(f"  source demo:  {_repo_rel(source_demo)}")
    print(f"  demo samples: {sample_count}, episodes: {episode_count}")
    for key, value in updates.items():
        print(f"  {key}: {value}")

    if args.dry_run:
        print("Dry run only; no files changed.")
        print("")
        print("Start training with:")
        print(training_command)
        return

    buffer_dir.mkdir(parents=True, exist_ok=True)
    output_dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_demo, demo_dest)
    replay_capacity = int(config.get("rlt_replay_capacity") or 50000)
    RLTReplayBuffer(capacity=replay_capacity).save(online_dest)
    _replace_config_keys(config_path, updates)

    source_hash = _sha256(source_demo)
    demo_hash = _sha256(demo_dest)
    if source_hash != demo_hash:
        raise RuntimeError(
            f"Copied demo buffer hash mismatch: source={source_hash} dest={demo_hash}"
        )

    print("")
    print(f"Created demo buffer:   {_repo_rel(demo_dest)}")
    print(f"Created online buffer: {_repo_rel(online_dest)}")
    print(f"Created output dir:    {_repo_rel(output_dest)}")
    print(f"Updated config:        {_repo_rel(config_path)}")
    print("")
    print("Start training with:")
    print(training_command)


if __name__ == "__main__":
    main()
