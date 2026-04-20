import torch
from torch.utils.data import Dataset


class DummyRobotDataset(Dataset):
    """
    Dummy dataset for a two-camera robot learning pipeline.

    Each sample contains:
    - overhead_image: fake RGB image tensor
    - wrist_image: fake RGB image tensor
    - robot_state: fake joint/state vector
    - instruction: task text
    - action: fake target action vector
    """

    def __init__(
        self,
        num_samples: int = 100,
        image_size: tuple[int, int, int] = (3, 64, 64),
        state_dim: int = 6,
        action_dim: int = 6,
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.instructions = [
            "hang the towel",
            "pick up the towel",
            "move the towel toward the bar",
            "align the towel for hanging",
        ]
        self.instruction_to_index = {instruction: idx for idx, instruction in enumerate(self.instructions)}

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        overhead_image = torch.rand(self.image_size, dtype=torch.float32)
        wrist_image = torch.rand(self.image_size, dtype=torch.float32)
        robot_state = torch.rand(self.state_dim, dtype=torch.float32)
        action = torch.rand(self.action_dim, dtype=torch.float32)

        instruction = self.instructions[idx % len(self.instructions)]
        instruction_id = torch.tensor(self.instruction_to_index[instruction], dtype=torch.long)

        return {
            "overhead_image": overhead_image,
            "wrist_image": wrist_image,
            "robot_state": robot_state,
            "instruction": instruction,
            "instruction_id": instruction_id,
            "action": action,
        }


if __name__ == "__main__":
    dataset = DummyRobotDataset(num_samples=5)
    sample = dataset[0]

    print("Sample keys:", sample.keys())
    print("Overhead image shape:", sample["overhead_image"].shape)
    print("Wrist image shape:", sample["wrist_image"].shape)
    print("Robot state shape:", sample["robot_state"].shape)
    print("Instruction:", sample["instruction"])
    print("Instruction ID:", sample["instruction_id"].item())
    print("Action shape:", sample["action"].shape)
