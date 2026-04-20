"""Dataset loader for real robot episodes stored as images plus CSV metadata."""

from pathlib import Path
import csv

import cv2
import torch
from torch.utils.data import Dataset


class RealRobotDataset(Dataset):
    def __init__(
        self,
        episode_dir: str | Path = "data/pilot_dataset/episode_1",
        metadata_filename: str = "metadata.csv",
    ):
        self.episode_dir = Path(episode_dir)
        self.metadata_path = self.episode_dir / metadata_filename

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        with self.metadata_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            self.rows = list(reader)

        self.has_base_image = any("base_frame" in row and row["base_frame"] for row in self.rows)
        instructions = sorted({row["instruction"] for row in self.rows})
        self.instruction_to_index = {
            instruction: index for index, instruction in enumerate(instructions)
        }
        self.index_to_instruction = {
            index: instruction for instruction, index in self.instruction_to_index.items()
        }

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        wrist_image_path = self.episode_dir / row["frame"]

        wrist_image = self._load_rgb_image(wrist_image_path)
        base_image = self._load_base_image(row)

        robot_state = torch.tensor(
            [float(row[f"state_{i}"]) for i in range(6)],
            dtype=torch.float32,
        )
        action = torch.tensor(
            [float(row[f"action_{i}"]) for i in range(6)],
            dtype=torch.float32,
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
            "instruction": instruction,
            "instruction_id": instruction_id,
            "action": action,
        }

    def _load_rgb_image(self, image_path: Path) -> torch.Tensor:
  1      image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
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


if __name__ == "__main__":
    dataset = RealRobotDataset()
    sample = dataset[0]

    print("Dataset length:", len(dataset))
    print("Image shape:", sample["wrist_image"].shape)
    print("Base image shape:", sample["base_image"].shape)
    print("Robot state shape:", sample["robot_state"].shape)
    print("Action shape:", sample["action"].shape)
    print("Instruction:", sample["instruction"])
