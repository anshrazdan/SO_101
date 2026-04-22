"""Scan the real-robot dataset for integrity issues and normalization statistics."""

from pathlib import Path
import argparse
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from python_src.datasets.real_robot_dataset import compute_dataset_statistics, scan_dataset_integrity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "data" / "pilot_dataset")
    parser.add_argument("--exclude-episode", action="append", default=[])
    return parser.parse_args()


def format_tensor(values) -> str:
    return "[" + ", ".join(f"{value:.6f}" for value in values.tolist()) + "]"


def main() -> None:
    args = parse_args()
    excluded_episodes = set(args.exclude_episode)

    report = scan_dataset_integrity(
        dataset_dir=args.dataset_dir,
        require_base_image=True,
    )
    print("Dataset integrity summary")
    print(f"  total_episode_dirs = {report.total_episode_dirs}")
    print(f"  valid_episodes = {report.valid_episodes}")
    print(f"  corrupted_episodes = {report.corrupted_episodes}")
    print(f"  metadata_missing_episodes = {report.metadata_missing_episodes}")
    print("  per_episode:")
    for episode in report.episodes:
        print(
            "   "
            f" {episode.episode_name}: rows={episode.total_rows} "
            f"valid_rows={episode.valid_rows} "
            f"missing_wrist_rows={episode.missing_wrist_rows} "
            f"missing_base_rows={episode.missing_base_rows} "
            f"missing_image_rows={episode.missing_image_rows} "
            f"metadata_missing={episode.metadata_missing}"
        )

    statistics = compute_dataset_statistics(
        dataset_dir=args.dataset_dir,
        require_base_image=True,
        excluded_episodes=excluded_episodes,
    )
    print()
    print("Dataset statistics")
    print(f"  total_valid_rows = {statistics.total_valid_rows}")
    print(f"  state_min = {format_tensor(statistics.state_min)}")
    print(f"  state_max = {format_tensor(statistics.state_max)}")
    print(f"  state_mean = {format_tensor(statistics.state_mean)}")
    print(f"  state_std = {format_tensor(statistics.state_std)}")
    print(f"  action_min = {format_tensor(statistics.action_min)}")
    print(f"  action_max = {format_tensor(statistics.action_max)}")
    print(f"  action_mean = {format_tensor(statistics.action_mean)}")
    print(f"  action_std = {format_tensor(statistics.action_std)}")
    print(f"  zero_action_fraction = {statistics.zero_action_fraction:.6f}")
    print(f"  action_variance_dominance = {format_tensor(statistics.action_variance_dominance)}")
    print("  action_magnitude_histogram:")
    for start, end, count in statistics.action_magnitude_histogram:
        print(f"    [{start:.4f}, {end:.4f}) -> {count}")


if __name__ == "__main__":
    main()
