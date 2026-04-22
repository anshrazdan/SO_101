"""Dataset loader and diagnostics for real robot episodes stored as images plus CSV metadata."""

from dataclasses import dataclass
from pathlib import Path
import csv
import math

import cv2
import torch
from torch.utils.data import ConcatDataset, Dataset


STATE_DIM = 6
ACTION_DIM = 6


@dataclass(frozen=True)
class EpisodeIntegrityReport:
    episode_name: str
    episode_dir: Path
    metadata_path: Path | None
    metadata_missing: bool
    total_rows: int
    valid_rows: int
    missing_wrist_rows: int
    missing_base_rows: int
    missing_image_rows: int
    has_base_images: bool

    @property
    def corrupted(self) -> bool:
        return self.metadata_missing or self.missing_image_rows > 0 or self.valid_rows == 0


@dataclass(frozen=True)
class DatasetIntegrityReport:
    total_episode_dirs: int
    valid_episodes: int
    corrupted_episodes: int
    metadata_missing_episodes: int
    episodes: list[EpisodeIntegrityReport]


@dataclass(frozen=True)
class DatasetStatistics:
    state_min: torch.Tensor
    state_max: torch.Tensor
    state_mean: torch.Tensor
    state_std: torch.Tensor
    action_min: torch.Tensor
    action_max: torch.Tensor
    action_mean: torch.Tensor
    action_std: torch.Tensor
    zero_action_fraction: float
    action_magnitude_histogram: list[tuple[float, float, int]]
    action_variance_dominance: torch.Tensor
    total_valid_rows: int

    def normalize_state(self, value: torch.Tensor) -> torch.Tensor:
        return normalize_tensor(value, self.state_mean, self.state_std)

    def denormalize_state(self, value: torch.Tensor) -> torch.Tensor:
        return denormalize_tensor(value, self.state_mean, self.state_std)

    def normalize_action(self, value: torch.Tensor) -> torch.Tensor:
        return normalize_tensor(value, self.action_mean, self.action_std)

    def denormalize_action(self, value: torch.Tensor) -> torch.Tensor:
        return denormalize_tensor(value, self.action_mean, self.action_std)


def safe_std(std: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.where(std < eps, torch.ones_like(std), std)


def normalize_tensor(value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (value - mean) / safe_std(std)


def denormalize_tensor(value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return value * safe_std(std) + mean


class RealRobotDataset(Dataset):
    def __init__(
        self,
        episode_dir: str | Path = "data/pilot_dataset/episode_1",
        metadata_filename: str = "metadata.csv",
        action_chunk_size: int = 10,
        instruction_to_index: dict[str, int] | None = None,
        use_base_image: bool | None = None,
        skip_invalid_rows: bool = False,
        require_base_image: bool = False,
        state_mean: torch.Tensor | None = None,
        state_std: torch.Tensor | None = None,
        action_mean: torch.Tensor | None = None,
        action_std: torch.Tensor | None = None,
    ):
        self.episode_dir = Path(episode_dir)
        self.metadata_path = self.episode_dir / metadata_filename
        self.action_chunk_size = action_chunk_size
        self.skip_invalid_rows = skip_invalid_rows
        self.require_base_image = require_base_image
        self.state_mean = state_mean
        self.state_std = state_std
        self.action_mean = action_mean
        self.action_std = action_std

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        with self.metadata_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            raw_rows = list(reader)

        self.has_base_image = any("base_frame" in row and row["base_frame"] for row in raw_rows)
        self.use_base_image = self.has_base_image if use_base_image is None else use_base_image
        self.rows = self._filter_rows(raw_rows)
        if instruction_to_index is None:
            instructions = sorted({row["instruction"] for row in self.rows})
            instruction_to_index = {
                instruction: index for index, instruction in enumerate(instructions)
            }

        self.instruction_to_index = instruction_to_index
        self.index_to_instruction = {
            index: instruction for instruction, index in self.instruction_to_index.items()
        }

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        wrist_image_path = self.episode_dir / row["frame"]

        wrist_image = self._load_rgb_image(wrist_image_path)
        base_image = self._load_base_image(row) if self.use_base_image else torch.empty(0, dtype=torch.float32)

        robot_state = self._load_state_tensor(row)
        action = self._load_action_tensor(row)
        action_chunk = torch.stack(
            [self._load_action_at(idx + offset) for offset in range(self.action_chunk_size)]
        )
        instruction = row["instruction"]
        instruction_id = torch.tensor(
            self.instruction_to_index[instruction],
            dtype=torch.long,
        )

        return {
            "wrist_image": wrist_image,
            "base_image": base_image,
            "robot_state": robot_state,
            "robot_state_raw": self._load_state_tensor(row, normalize=False),
            "instruction": instruction,
            "instruction_id": instruction_id,
            "action": action,
            "action_raw": self._load_action_tensor(row, normalize=False),
            "action_chunk": action_chunk,
            "action_chunk_raw": torch.stack(
                [self._load_action_at(idx + offset, normalize=False) for offset in range(self.action_chunk_size)]
            ),
        }

    def _filter_rows(self, rows: list[dict[str, str]]) -> list[dict[str, str]]:
        if not self.skip_invalid_rows:
            return rows

        return [row for row in rows if self._row_is_valid(row)]

    def _row_is_valid(self, row: dict[str, str]) -> bool:
        wrist_image_path = self.episode_dir / row["frame"]
        if not wrist_image_path.exists():
            return False

        if self.require_base_image or self.use_base_image:
            base_frame = row.get("base_frame", "")
            if not base_frame:
                return False
            if not (self.episode_dir / base_frame).exists():
                return False

        return True

    def _load_state_tensor(self, row: dict[str, str], normalize: bool = True) -> torch.Tensor:
        state = torch.tensor(
            [float(row[f"state_{i}"]) for i in range(STATE_DIM)],
            dtype=torch.float32,
        )
        if normalize and self.state_mean is not None and self.state_std is not None:
            state = normalize_tensor(state, self.state_mean, self.state_std)
        return state

    def _load_action_tensor(self, row: dict[str, str], normalize: bool = True) -> torch.Tensor:
        action = torch.tensor(
            [float(row[f"action_{i}"]) for i in range(ACTION_DIM)],
            dtype=torch.float32,
        )
        if normalize and self.action_mean is not None and self.action_std is not None:
            action = normalize_tensor(action, self.action_mean, self.action_std)
        return action

    def _load_action_at(self, idx: int, normalize: bool = True) -> torch.Tensor:
        row = self.rows[min(idx, len(self.rows) - 1)]
        return self._load_action_tensor(row, normalize=normalize)

    def _load_rgb_image(self, image_path: Path) -> torch.Tensor:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found or unreadable: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

    def _load_base_image(self, row: dict[str, str]) -> torch.Tensor:
        base_frame = row.get("base_frame", "")
        if not base_frame:
            return torch.empty(0, dtype=torch.float32)

        base_image_path = self.episode_dir / base_frame
        return self._load_rgb_image(base_image_path)


def find_episode_dirs(dataset_dir: str | Path = "data/pilot_dataset") -> list[Path]:
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    return sorted(
        [path for path in dataset_dir.iterdir() if path.is_dir() and path.name.startswith("episode_")],
        key=lambda path: (
            0,
            int(path.name.split("_")[-1]),
        ) if path.name.split("_")[-1].isdigit() else (1, path.name),
    )


def preferred_metadata_filename(episode_dir: Path) -> str:
    if (episode_dir / "metadata_clean.csv").exists():
        return "metadata_clean.csv"
    return "metadata.csv"


def build_instruction_index(
    episode_dirs: list[Path],
    metadata_filename: str | None = None,
) -> dict[str, int]:
    instructions: set[str] = set()

    for episode_dir in episode_dirs:
        filename = metadata_filename or preferred_metadata_filename(episode_dir)
        metadata_path = episode_dir / filename
        if not metadata_path.exists():
            continue

        with metadata_path.open("r", newline="", encoding="utf-8") as csv_file:
            for row in csv.DictReader(csv_file):
                instructions.add(row["instruction"])

    if not instructions:
        raise ValueError("No instructions found in dataset metadata.")

    return {
        instruction: index
        for index, instruction in enumerate(sorted(instructions))
    }


def metadata_has_base_image(metadata_path: Path) -> bool:
    with metadata_path.open("r", newline="", encoding="utf-8") as csv_file:
        return any(
            "base_frame" in row and row["base_frame"]
            for row in csv.DictReader(csv_file)
        )


def scan_episode_integrity(
    episode_dir: Path,
    metadata_filename: str | None = None,
    require_base_image: bool = True,
) -> EpisodeIntegrityReport:
    filename = metadata_filename or preferred_metadata_filename(episode_dir)
    metadata_path = episode_dir / filename
    if not metadata_path.exists():
        return EpisodeIntegrityReport(
            episode_name=episode_dir.name,
            episode_dir=episode_dir,
            metadata_path=None,
            metadata_missing=True,
            total_rows=0,
            valid_rows=0,
            missing_wrist_rows=0,
            missing_base_rows=0,
            missing_image_rows=0,
            has_base_images=False,
        )

    with metadata_path.open("r", newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    missing_wrist_rows = 0
    missing_base_rows = 0
    valid_rows = 0
    has_base_images = any(row.get("base_frame", "") for row in rows)

    for row in rows:
        wrist_exists = (episode_dir / row["frame"]).exists()
        base_frame = row.get("base_frame", "")
        if require_base_image or has_base_images:
            base_exists = bool(base_frame) and (episode_dir / base_frame).exists()
        else:
            base_exists = True

        if not wrist_exists:
            missing_wrist_rows += 1
        if not base_exists:
            missing_base_rows += 1

        if wrist_exists and base_exists:
            valid_rows += 1

    missing_image_rows = sum(
        1
        for row in rows
        if not (episode_dir / row["frame"]).exists()
        or (
            (require_base_image or has_base_images)
            and (not row.get("base_frame", "") or not (episode_dir / row["base_frame"]).exists())
        )
    )

    return EpisodeIntegrityReport(
        episode_name=episode_dir.name,
        episode_dir=episode_dir,
        metadata_path=metadata_path,
        metadata_missing=False,
        total_rows=len(rows),
        valid_rows=valid_rows,
        missing_wrist_rows=missing_wrist_rows,
        missing_base_rows=missing_base_rows,
        missing_image_rows=missing_image_rows,
        has_base_images=has_base_images,
    )


def scan_dataset_integrity(
    dataset_dir: str | Path = "data/pilot_dataset",
    metadata_filename: str | None = None,
    require_base_image: bool = True,
) -> DatasetIntegrityReport:
    episode_dirs = find_episode_dirs(dataset_dir)
    reports = [
        scan_episode_integrity(
            episode_dir,
            metadata_filename=metadata_filename,
            require_base_image=require_base_image,
        )
        for episode_dir in episode_dirs
    ]
    corrupted_episodes = sum(report.corrupted for report in reports)
    metadata_missing_episodes = sum(report.metadata_missing for report in reports)
    valid_episodes = sum(not report.corrupted for report in reports)
    return DatasetIntegrityReport(
        total_episode_dirs=len(episode_dirs),
        valid_episodes=valid_episodes,
        corrupted_episodes=corrupted_episodes,
        metadata_missing_episodes=metadata_missing_episodes,
        episodes=reports,
    )


def compute_dataset_statistics(
    dataset_dir: str | Path = "data/pilot_dataset",
    metadata_filename: str | None = None,
    require_base_image: bool = True,
    excluded_episodes: set[str] | None = None,
    included_episodes: set[str] | None = None,
) -> DatasetStatistics:
    excluded_episodes = excluded_episodes or set()
    included_episodes = included_episodes or set()
    report = scan_dataset_integrity(
        dataset_dir=dataset_dir,
        metadata_filename=metadata_filename,
        require_base_image=require_base_image,
    )

    valid_rows: list[dict[str, str]] = []
    for episode_report in report.episodes:
        if included_episodes and episode_report.episode_name not in included_episodes:
            continue
        if episode_report.episode_name in excluded_episodes or episode_report.metadata_missing:
            continue
        if episode_report.corrupted:
            continue

        assert episode_report.metadata_path is not None
        with episode_report.metadata_path.open("r", newline="", encoding="utf-8") as csv_file:
            rows = list(csv.DictReader(csv_file))

        episode_dir = episode_report.episode_dir
        for row in rows:
            wrist_exists = (episode_dir / row["frame"]).exists()
            base_frame = row.get("base_frame", "")
            base_exists = (
                bool(base_frame)
                and (episode_dir / base_frame).exists()
            ) if require_base_image or episode_report.has_base_images else True
            if wrist_exists and base_exists:
                valid_rows.append(row)

    if not valid_rows:
        raise ValueError("No valid rows found for statistics.")

    state = torch.tensor(
        [[float(row[f"state_{i}"]) for i in range(STATE_DIM)] for row in valid_rows],
        dtype=torch.float32,
    )
    action = torch.tensor(
        [[float(row[f"action_{i}"]) for i in range(ACTION_DIM)] for row in valid_rows],
        dtype=torch.float32,
    )
    action_magnitude = torch.linalg.vector_norm(action, dim=1)
    zero_action_fraction = (
        torch.isclose(action.abs().sum(dim=1), torch.zeros(1, dtype=action.dtype)).float().mean().item()
    )

    hist = torch.histc(
        action_magnitude,
        bins=10,
        min=float(action_magnitude.min()),
        max=float(action_magnitude.max()) if action_magnitude.numel() > 0 else 1.0,
    )
    histogram: list[tuple[float, float, int]] = []
    if hist.numel() > 0:
        min_mag = float(action_magnitude.min())
        max_mag = float(action_magnitude.max())
        if math.isclose(min_mag, max_mag):
            histogram.append((min_mag, max_mag, int(hist.sum().item())))
        else:
            bucket_width = (max_mag - min_mag) / hist.numel()
            for bucket_index, bucket_count in enumerate(hist.tolist()):
                start = min_mag + bucket_index * bucket_width
                end = start + bucket_width
                histogram.append((start, end, int(bucket_count)))

    action_std = action.std(dim=0, unbiased=False)
    action_variance_dominance = action_std.pow(2) / action_std.pow(2).sum()

    return DatasetStatistics(
        state_min=state.min(dim=0).values,
        state_max=state.max(dim=0).values,
        state_mean=state.mean(dim=0),
        state_std=state.std(dim=0, unbiased=False),
        action_min=action.min(dim=0).values,
        action_max=action.max(dim=0).values,
        action_mean=action.mean(dim=0),
        action_std=action_std,
        zero_action_fraction=zero_action_fraction,
        action_magnitude_histogram=histogram,
        action_variance_dominance=action_variance_dominance,
        total_valid_rows=state.size(0),
    )


class RealRobotEpisodeCollection(ConcatDataset):
    def __init__(
        self,
        dataset_dir: str | Path = "data/pilot_dataset",
        metadata_filename: str | None = None,
        action_chunk_size: int = 10,
        skip_invalid_rows: bool = False,
        skip_corrupted_episodes: bool = False,
        excluded_episodes: set[str] | None = None,
        require_base_image: bool = True,
        normalization_stats: DatasetStatistics | None = None,
        normalize_state: bool = False,
        normalize_action: bool = False,
    ):
        episode_dirs = find_episode_dirs(dataset_dir)
        if not episode_dirs:
            raise FileNotFoundError(f"No episode folders found in {dataset_dir}")
        excluded_episodes = excluded_episodes or set()

        self.instruction_to_index = build_instruction_index(episode_dirs, metadata_filename)
        self.index_to_instruction = {
            index: instruction for instruction, index in self.instruction_to_index.items()
        }
        self.integrity_report = scan_dataset_integrity(
            dataset_dir=dataset_dir,
            metadata_filename=metadata_filename,
            require_base_image=require_base_image,
        )

        usable_episodes = []
        for episode_report in self.integrity_report.episodes:
            if episode_report.episode_name in excluded_episodes:
                continue
            if episode_report.metadata_missing:
                continue
            if skip_corrupted_episodes and episode_report.corrupted:
                continue

            assert episode_report.metadata_path is not None
            usable_episodes.append(
                (
                    episode_report.episode_dir,
                    episode_report.metadata_path.name,
                    episode_report.metadata_path,
                    episode_report,
                )
            )

        if not usable_episodes:
            raise FileNotFoundError(f"No usable metadata files found in {dataset_dir}")

        self.has_base_image = all(
            metadata_has_base_image(metadata_path)
            for _, _, metadata_path, _ in usable_episodes
        )

        datasets = []
        for episode_dir, filename, _, episode_report in usable_episodes:
            datasets.append(
                RealRobotDataset(
                    episode_dir=episode_dir,
                    metadata_filename=filename,
                    action_chunk_size=action_chunk_size,
                    instruction_to_index=self.instruction_to_index,
                    use_base_image=self.has_base_image,
                    skip_invalid_rows=skip_invalid_rows,
                    require_base_image=require_base_image or episode_report.has_base_images,
                    state_mean=normalization_stats.state_mean if normalization_stats and normalize_state else None,
                    state_std=normalization_stats.state_std if normalization_stats and normalize_state else None,
                    action_mean=normalization_stats.action_mean if normalization_stats and normalize_action else None,
                    action_std=normalization_stats.action_std if normalization_stats and normalize_action else None,
                )
            )

        self.normalization_stats = normalization_stats
        super().__init__(datasets)


if __name__ == "__main__":
    dataset = RealRobotDataset()
    sample = dataset[0]

    print("Dataset length:", len(dataset))
    print("Image shape:", sample["wrist_image"].shape)
    print("Base image shape:", sample["base_image"].shape)
    print("Robot state shape:", sample["robot_state"].shape)
    print("Action shape:", sample["action"].shape)
    print("Action chunk shape:", sample["action_chunk"].shape)
    print("Instruction:", sample["instruction"])
