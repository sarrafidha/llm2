import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set


def _load_dataset(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Accept both list of samples or {"question": [...]} format
    if isinstance(data, dict) and "question" in data:
        data = data["question"]
    if not isinstance(data, list):
        raise ValueError("Unsupported dataset format: expected a list or a dict with 'question'.")
    return data


def _save_dataset(path: str, samples: List[Dict]) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(path_obj, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False)


def _select_unique_img_ids(samples: List[Dict], limit: int, seed: int, max_samples_per_image: int) -> List[Dict]:
    # Group samples by img_id
    img_id_to_samples: Dict[str, List[Dict]] = {}
    for s in samples:
        img_id = s.get("img_id")
        if img_id is None:
            # Skip entries without image id (not compatible with ChangeChat multi-image training)
            continue
        img_id_to_samples.setdefault(img_id, []).append(s)

    # Shuffle image ids for a diverse subset
    rng = random.Random(seed)
    unique_ids: List[str] = list(img_id_to_samples.keys())
    rng.shuffle(unique_ids)
    selected_ids = unique_ids[:limit]

    # For each selected id, pick up to max_samples_per_image examples
    filtered: List[Dict] = []
    for img_id in selected_ids:
        group = img_id_to_samples[img_id]
        rng.shuffle(group)
        filtered.extend(group[:max_samples_per_image])
    return filtered


def _remove_arg(args_list: List[str], flag: str) -> List[str]:
    # Remove occurrences of a flag and its value (if present)
    cleaned: List[str] = []
    i = 0
    while i < len(args_list):
        if args_list[i] == flag:
            # Skip flag and next value if exists
            i += 2 if (i + 1) < len(args_list) and not args_list[i + 1].startswith("--") else 1
        else:
            cleaned.append(args_list[i])
            i += 1
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter dataset to N unique image IDs and run training.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the original training JSON.")
    parser.add_argument("--limit", type=int, default=1000, help="Number of unique image IDs to keep.")
    parser.add_argument(
        "--max_samples_per_image", type=int, default=1, help="Max samples to keep per unique image ID."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument(
        "--output_data_path",
        type=str,
        default="",
        help="Optional output JSON path. Defaults to <data_path stem>_limit<limit>.json in the same folder.",
    )

    # Capture additional args for the underlying trainer
    known_args, passthrough = parser.parse_known_args()

    original_data_path = known_args.data_path
    limit = known_args.limit
    max_per_img = known_args.max_samples_per_image
    seed = known_args.seed

    if known_args.output_data_path:
        filtered_path = known_args.output_data_path
    else:
        p = Path(original_data_path)
        filtered_path = str(p.with_name(f"{p.stem}_limit{limit}{p.suffix}"))

    # Build filtered dataset
    samples = _load_dataset(original_data_path)
    filtered_samples = _select_unique_img_ids(samples, limit=limit, seed=seed, max_samples_per_image=max_per_img)
    if len(filtered_samples) == 0:
        raise ValueError("No samples selected. Ensure your dataset contains 'img_id' fields.")
    _save_dataset(filtered_path, filtered_samples)

    # Prepare argv for the underlying trainer (train.py) with the new data_path
    forwarded_args: List[str] = list(passthrough)
    forwarded_args = _remove_arg(forwarded_args, "--data_path")
    forwarded_args.extend(["--data_path", filtered_path])

    # Run the standard trainer entrypoint without FlashAttention
    from changechat.train.train import train as run_train

    old_argv = list(sys.argv)
    try:
        sys.argv = [old_argv[0]] + forwarded_args
        run_train()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main() 