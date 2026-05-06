"""Train an ACT-style action chunking model from images, robot state, and instructions."""

from pathlib import Path
import sys
import csv

import torch
from torch import nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from python_src.datasets.real_robot_dataset import RealRobotEpisodeCollection


ACTION_CHUNK_SIZE = 10


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


class ActionChunkingTransformer(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_instructions: int,
        use_base_image: bool = False,
        action_chunk_size: int = ACTION_CHUNK_SIZE,
        hidden_dim: int = 128,
        image_feature_dim: int = 64,
        instruction_embedding_dim: int = 16,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        self.use_base_image = use_base_image
        self.action_chunk_size = action_chunk_size
        self.wrist_encoder = ImageEncoder(output_dim=image_feature_dim)
        self.base_encoder = ImageEncoder(output_dim=image_feature_dim) if use_base_image else None
        self.instruction_embedding = nn.Embedding(
            num_embeddings=num_instructions,
            embedding_dim=instruction_embedding_dim,
        )
        self.wrist_projection = nn.Linear(image_feature_dim, hidden_dim)
        self.base_projection = nn.Linear(image_feature_dim, hidden_dim) if use_base_image else None
        self.state_projection = nn.Linear(state_dim, hidden_dim)
        self.instruction_projection = nn.Linear(instruction_embedding_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.observation_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.action_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )
        self.action_queries = nn.Parameter(torch.randn(action_chunk_size, hidden_dim) * 0.02)
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(
        self,
        wrist_image: torch.Tensor,
        base_image: torch.Tensor | None,
        robot_state: torch.Tensor,
        instruction_id: torch.Tensor,
    ) -> torch.Tensor:
        wrist_token = self.wrist_projection(self.wrist_encoder(wrist_image)).unsqueeze(1)
        state_token = self.state_projection(robot_state).unsqueeze(1)
        instruction_features = self.instruction_embedding(instruction_id)
        instruction_token = self.instruction_projection(instruction_features).unsqueeze(1)

        observation_tokens = [wrist_token, state_token, instruction_token]
        if self.use_base_image:
            if base_image is None or base_image.numel() == 0:
                raise ValueError("Model expects base_image input, but none was provided.")
            base_token = self.base_projection(self.base_encoder(base_image)).unsqueeze(1)
            observation_tokens.insert(1, base_token)

        memory = self.observation_encoder(torch.cat(observation_tokens, dim=1))
        action_queries = self.action_queries.unsqueeze(0).expand(robot_state.size(0), -1, -1)
        decoded_actions = self.action_decoder(action_queries, memory)
        return self.action_head(decoded_actions)


RealRobotActionNet = ActionChunkingTransformer
WristImageStateToActionNet = ActionChunkingTransformer


def main() -> None:
    dataset = RealRobotEpisodeCollection(
        PROJECT_ROOT / "data" / "pilot_dataset",
        action_chunk_size=ACTION_CHUNK_SIZE,
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    loss_csv_path = PROJECT_ROOT / "data" / "real_training_loss.csv"
    model_path = PROJECT_ROOT / "data" / "act_model.pth"
    loss_csv_path.parent.mkdir(parents=True, exist_ok=True)

    model = ActionChunkingTransformer(
        state_dim=6,
        action_dim=6,
        num_instructions=len(dataset.instruction_to_index),
        use_base_image=dataset.has_base_image,
        action_chunk_size=ACTION_CHUNK_SIZE,
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    num_epochs = 50
    epoch_losses: list[tuple[int, float]] = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for step, batch in enumerate(dataloader, start=1):
            wrist_image = batch["wrist_image"]
            base_image = batch["base_image"]
            robot_state = batch["robot_state"]
            instruction_id = batch["instruction_id"]
            target_action_chunk = batch["action_chunk"]

            predicted_action_chunk = model(wrist_image, base_image, robot_state, instruction_id)
            loss = criterion(predicted_action_chunk, target_action_chunk)

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

    checkpoint = {}
    checkpoint["model_state_dict"] = model.state_dict()
    checkpoint["state_mean"] = dataset.state_mean
    checkpoint["state_std"] = dataset.state_std
    checkpoint["action_mean"] = dataset.action_mean
    checkpoint["action_std"] = dataset.action_std
    checkpoint["instruction_to_index"] = dataset.instruction_to_index

    torch.save(checkpoint, model_path)
    print(f"saved model weights to {model_path}")


if __name__ == "__main__":
    main()
