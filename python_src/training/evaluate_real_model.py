"""Evaluate the trained ACT model and export predicted versus true action chunks."""

from pathlib import Path
import sys
import csv

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from python_src.datasets.real_robot_dataset import RealRobotEpisodeCollection
from python_src.training.train_real import ACTION_CHUNK_SIZE, ActionChunkingTransformer


def main() -> None:
    dataset = RealRobotEpisodeCollection(
        PROJECT_ROOT / "data" / "pilot_dataset",
        action_chunk_size=ACTION_CHUNK_SIZE,
    )
    model_path = PROJECT_ROOT / "data" / "act_model.pth"
    predictions_path = PROJECT_ROOT / "data" / "act_model_predictions.csv"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {model_path}. Run train_real.py first."
        )

    model = ActionChunkingTransformer(
        state_dim=6,
        action_dim=6,
        num_instructions=len(dataset.instruction_to_index),
        use_base_image=dataset.has_base_image,
        action_chunk_size=ACTION_CHUNK_SIZE,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    num_samples = min(20, len(dataset))

    with predictions_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        header = ["sample_idx", "instruction"]
        header += [f"true_action_{i}" for i in range(6)]
        header += [f"pred_action_{i}" for i in range(6)]
        header += [
            f"pred_chunk_t{step}_action_{joint}"
            for step in range(ACTION_CHUNK_SIZE)
            for joint in range(6)
        ]
        writer.writerow(header)

        with torch.no_grad():
            for idx in range(num_samples):
                sample = dataset[idx]
                predicted_action_chunk = model(
                    sample["wrist_image"].unsqueeze(0),
                    sample["base_image"].unsqueeze(0),
                    sample["robot_state"].unsqueeze(0),
                    sample["instruction_id"].unsqueeze(0),
                ).squeeze(0)
                predicted_action = predicted_action_chunk[0]

                true_action = sample["action"].tolist()
                predicted_action_list = predicted_action.tolist()
                predicted_action_chunk_list = predicted_action_chunk.flatten().tolist()

                print(f"sample {idx}")
                print(f"  instruction: {sample['instruction']}")
                print(f"  ground truth action: {true_action}")
                print(f"  predicted action:    {predicted_action_list}")
                print()

                row = [idx, sample["instruction"]]
                row += true_action
                row += predicted_action_list
                row += predicted_action_chunk_list
                writer.writerow(row)

    print(f"saved predictions to {predictions_path}")

if __name__ == "__main__":
    main()
