"""Dataset loader for real robot episodes stored as images plus CSV metadata."""

from pathlib import Path
import csv

import cv2
import torch
from torch.utils.data import ConcatDataset, Dataset


class RealRobotDataset(Dataset):
    def __init__(
        self,
        episode_dir: str | Path = "data/pilot_dataset/episode_1",
        metadata_filename: str = "metadata.csv",
        action_chunk_size: int = 10,
        instruction_to_index: dict[str, int] | None = None,
        use_base_image: bool | None = None,
        state_mean: torch.Tensor | None = None,
        state_std: torch.Tensor | None = None,
        action_mean: torch.Tensor | None = None,
        action_std: torch.Tensor | None = None,
    ):
        self.episode_dir = Path(episode_dir)
        self.metadata_path = self.episode_dir / metadata_filename
        self.action_chunk_size = action_chunk_size

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        with self.metadata_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            self.rows = list(reader)

        self.has_base_image = any("base_frame" in row and row["base_frame"] for row in self.rows)
        self.use_base_image = self.has_base_image if use_base_image is None else use_base_image
        if instruction_to_index is None:
            instructions = sorted({row["instruction"] for row in self.rows})
            instruction_to_index = {
                instruction: index for index, instruction in enumerate(instructions)
            }

        self.instruction_to_index = instruction_to_index
        self.index_to_instruction = {
            index: instruction for instruction, index in self.instruction_to_index.items()
        }

        if state_mean is None:
            self.state_mean = torch.zeros(6, dtype=torch.float32)
        else:
            self.state_mean = state_mean

        if state_std is None:
            self.state_std = torch.ones(6, dtype=torch.float32)
        else:
            self.state_std = state_std

        if action_mean is None:
            self.action_mean = torch.zeros(6, dtype=torch.float32)
        else:
            self.action_mean = action_mean

        if action_std is None:
            self.action_std = torch.ones(6, dtype=torch.float32)
        else:
            self.action_std = action_std

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        wrist_image_path = self.episode_dir / row["frame"]

        wrist_image = self._load_rgb_image(wrist_image_path)
        base_image = self._load_base_image(row) if self.use_base_image else torch.empty(0, dtype=torch.float32)

        robot_state = torch.tensor(
            [float(row[f"state_{i}"]) for i in range(6)],
            dtype=torch.float32,
        )
        action = torch.tensor(
            [float(row[f"action_{i}"]) for i in range(6)],
            dtype=torch.float32,
        )
        action_chunk = torch.stack(
            [self._load_action_at(idx + offset) for offset in range(self.action_chunk_size)]
        )

        robot_state = (robot_state - self.state_mean) / self.state_std
        action = (action - self.action_mean) / self.action_std
        action_chunk = (action_chunk - self.action_mean) / self.action_std
        instruction = row["instruction"]
        instruction_id = torch.tensor(
            self.instruction_to_index[instruction],
            dtype=torch.long,
        )

        return {
            "wrist_image": wrist_image,
            "base_image": base_image,
            "robot_state": robot_state,
            "instruction": instruction,
            "instruction_id": instruction_id,
            "action": action,
            "action_chunk": action_chunk,
        }

    def _load_action_at(self, idx: int) -> torch.Tensor:
        row = self.rows[min(idx, len(self.rows) - 1)]
        return torch.tensor(
            [float(row[f"action_{i}"]) for i in range(6)],
            dtype=torch.float32,
        )

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


def compute_state_action_stats(episode_dirs, metadata_filename=None):
    state_values = []
    action_values = []
    for i in range(6):
        state_values.append([])
        action_values.append([])

    for episode_dir in episode_dirs:
        if metadata_filename is None:
            filename = preferred_metadata_filename(episode_dir)
        else:
            filename = metadata_filename

        metadata_path = episode_dir / filename
        if not metadata_path.exists():
            continue

        with metadata_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                for i in range(6):
                    state_values[i].append(float(row[f"state_{i}"]))
                    action_values[i].append(float(row[f"action_{i}"]))

    state_tensor = torch.tensor(state_values, dtype=torch.float32)
    action_tensor = torch.tensor(action_values, dtype=torch.float32)

    state_mean = state_tensor.mean(dim=1)
    state_std = state_tensor.std(dim=1)
    action_mean = action_tensor.mean(dim=1)
    action_std = action_tensor.std(dim=1)

    state_std = state_std.clamp(min=1e-6)
    action_std = action_std.clamp(min=1e-6)

    stats = {}
    stats["state_mean"] = state_mean
    stats["state_std"] = state_std
    stats["action_mean"] = action_mean
    stats["action_std"] = action_std
    return stats


def metadata_has_base_image(metadata_path: Path) -> bool:
    with metadata_path.open("r", newline="", encoding="utf-8") as csv_file:
        return any(
            "base_frame" in row and row["base_frame"]
            for row in csv.DictReader(csv_file)
        )


class RealRobotEpisodeCollection(ConcatDataset):
    def __init__(
        self,
        dataset_dir: str | Path = "data/pilot_dataset",
        metadata_filename: str | None = None,
        action_chunk_size: int = 10,
    ):
        episode_dirs = find_episode_dirs(dataset_dir)
        if not episode_dirs:
            raise FileNotFoundError(f"No episode folders found in {dataset_dir}")

        self.instruction_to_index = build_instruction_index(episode_dirs, metadata_filename)
        self.index_to_instruction = {
            index: instruction for instruction, index in self.instruction_to_index.items()
        }

        usable_episodes = []
        for episode_dir in episode_dirs:
            filename = metadata_filename or preferred_metadata_filename(episode_dir)
            metadata_path = episode_dir / filename
            if not metadata_path.exists():
                continue

            usable_episodes.append((episode_dir, filename, metadata_path))

        if not usable_episodes:
            raise FileNotFoundError(f"No usable metadata files found in {dataset_dir}")

        self.has_base_image = all(
            metadata_has_base_image(metadata_path)
            for _, _, metadata_path in usable_episodes
        )

        stats = compute_state_action_stats(episode_dirs, metadata_filename)
        self.state_mean = stats["state_mean"]
        self.state_std = stats["state_std"]
        self.action_mean = stats["action_mean"]
        self.action_std = stats["action_std"]

        datasets = []
        for episode_dir, filename, _ in usable_episodes:
            datasets.append(
                RealRobotDataset(
                    episode_dir=episode_dir,
                    metadata_filename=filename,
                    action_chunk_size=action_chunk_size,
                    instruction_to_index=self.instruction_to_index,
                    use_base_image=self.has_base_image,
                    state_mean=self.state_mean,
                    state_std=self.state_std,
                    action_mean=self.action_mean,
                    action_std=self.action_std,
                )
            )

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
