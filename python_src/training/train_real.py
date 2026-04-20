"""Train the real robot action model from wrist images, robot state, and instructions."""

from pathlib import Path
import sys
import csv

import torch
from torch import nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from python_src.datasets.real_robot_dataset import RealRobotDataset


class ImageEncoder(nn.Module):
    def __init__(self, output_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, output_dim),
            nn.ReLU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.encoder(image)


class RealRobotActionNet(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_instructions: int,
        use_base_image: bool = False,
        image_feature_dim: int = 64,
        instruction_embedding_dim: int = 16,
    ):
        super().__init__()
        self.use_base_image = use_base_image
        self.wrist_encoder = ImageEncoder(output_dim=image_feature_dim)
        self.base_encoder = ImageEncoder(output_dim=image_feature_dim) if use_base_image else None
        self.instruction_embedding = nn.Embedding(
            num_embeddings=num_instructions,
            embedding_dim=instruction_embedding_dim,
        )
        fusion_input_dim = image_feature_dim + state_dim + instruction_embedding_dim
        if use_base_image:
            fusion_input_dim += image_feature_dim

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 64),
            nn.ReLU(),
        )
        self.action_head = nn.Sequential(
            nn.Linear(64, action_dim),
        )

    def forward(
        self,
        wrist_image: torch.Tensor,
        base_image: torch.Tensor | None,
        robot_state: torch.Tensor,
        instruction_id: torch.Tensor,
    ) -> torch.Tensor:
        wrist_features = self.wrist_encoder(wrist_image)
        instruction_features = self.instruction_embedding(instruction_id)

        features = [wrist_features, robot_state, instruction_features]
        if self.use_base_image:
            if base_image is None or base_image.numel() == 0:
                raise ValueError("Model expects base_image input, but none was provided.")
            features.insert(1, self.base_encoder(base_image))

        combined_features = torch.cat(features, dim=1)
        fused_features = self.fusion(combined_features)
        return self.action_head(fused_features)


WristImageStateToActionNet = RealRobotActionNet


def main() -> None:
    dataset = RealRobotDataset(
        PROJECT_ROOT / "data" / "pilot_dataset" / "episode_1",
        metadata_filename="metadata_clean.csv",
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    loss_csv_path = PROJECT_ROOT / "data" / "real_training_loss.csv"
    model_path = PROJECT_ROOT / "data" / "real_model.pth"
    loss_csv_path.parent.mkdir(parents=True, exist_ok=True)

    model = RealRobotActionNet(
        state_dim=6,
        action_dim=6,
        num_instructions=len(dataset.instruction_to_index),
        use_base_image=dataset.has_base_image,
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 5
    epoch_losses: list[tuple[int, float]] = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for step, batch in enumerate(dataloader, start=1):
            wrist_image = batch["wrist_image"]
            base_image = batch["base_image"]
            robot_state = batch["robot_state"]
            instruction_id = batch["instruction_id"]
            target_action = batch["action"]

            predicted_action = model(wrist_image, base_image, robot_state, instruction_id)
            loss = criterion(predicted_action, target_action)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            print(f"epoch {epoch + 1}/{num_epochs} step {step}/{len(dataloader)} loss={loss.item():.4f}")

        average_loss = epoch_loss / len(dataloader)
        epoch_losses.append((epoch + 1, average_loss))
        print(f"epoch {epoch + 1} average_loss={average_loss:.4f}")

    with loss_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["epoch", "average_loss"])
        writer.writerows(epoch_losses)

    torch.save(model.state_dict(), model_path)
    print(f"saved model weights to {model_path}")


if __name__ == "__main__":
    main()
