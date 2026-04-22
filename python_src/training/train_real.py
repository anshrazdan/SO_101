"""Train an ACT-style action chunking model from images, robot state, and instructions."""

from pathlib import Path
import argparse
import random
import sys
import csv

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from python_src.datasets.real_robot_dataset import (
    DatasetStatistics,
    RealRobotEpisodeCollection,
    compute_dataset_statistics,
)


ACTION_CHUNK_SIZE = 10
BASELINE_NAME = "normalized_filtered_baseline"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "data" / "pilot_dataset")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--debug-batches", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-invalid-rows", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-corrupted-episodes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normalize-state", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normalize-action", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exclude-episode", action="append", default=[])
    parser.add_argument("--overfit-episodes", type=int, default=0)
    return parser.parse_args()


def format_tensor_list(values: torch.Tensor) -> str:
    return "[" + ", ".join(f"{value:.6f}" for value in values.tolist()) + "]"


def compute_loss_breakdown(
    predicted_action_chunk: torch.Tensor,
    target_action_chunk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    squared_error = (predicted_action_chunk - target_action_chunk).pow(2)
    per_joint_loss = squared_error.mean(dim=(0, 1))
    total_loss = per_joint_loss.mean()
    return total_loss, per_joint_loss


def print_statistics(statistics: DatasetStatistics) -> None:
    print("Training-split normalization statistics")
    print(f"  total_valid_rows = {statistics.total_valid_rows}")
    print(f"  state_mean = {format_tensor_list(statistics.state_mean)}")
    print(f"  state_std = {format_tensor_list(statistics.state_std)}")
    print(f"  action_mean = {format_tensor_list(statistics.action_mean)}")
    print(f"  action_std = {format_tensor_list(statistics.action_std)}")
    print(f"  zero_action_fraction = {statistics.zero_action_fraction:.6f}")
    print(f"  action_variance_dominance = {format_tensor_list(statistics.action_variance_dominance)}")


def build_episode_splits(
    dataset: RealRobotEpisodeCollection,
    val_ratio: float,
    seed: int,
    overfit_episodes: int = 0,
) -> tuple[list[int], list[int]]:
    episode_indices = list(range(len(dataset.datasets)))
    if overfit_episodes > 0:
        episode_indices = episode_indices[:overfit_episodes]

    if len(episode_indices) < 2:
        raise ValueError("Need at least 2 episodes to create a train/validation split.")

    rng = random.Random(seed)
    shuffled = episode_indices[:]
    rng.shuffle(shuffled)

    val_episode_count = max(1, round(len(shuffled) * val_ratio))
    val_episode_count = min(val_episode_count, len(shuffled) - 1)
    val_episode_indices = sorted(shuffled[:val_episode_count])
    train_episode_indices = sorted(shuffled[val_episode_count:])
    return train_episode_indices, val_episode_indices


def build_row_subset_from_episode_indices(
    dataset: RealRobotEpisodeCollection,
    episode_indices: list[int],
) -> Subset:
    row_indices: list[int] = []
    offset = 0
    for dataset_index, child_dataset in enumerate(dataset.datasets):
        child_length = len(child_dataset)
        if dataset_index in episode_indices:
            row_indices.extend(range(offset, offset + child_length))
        offset += child_length
    return Subset(dataset, row_indices)


def get_episode_names(
    dataset: RealRobotEpisodeCollection,
    episode_indices: list[int],
) -> list[str]:
    return [dataset.datasets[index].episode_dir.name for index in episode_indices]


def print_split_summary(
    train_episode_names: list[str],
    val_episode_names: list[str],
    train_subset: Subset,
    val_subset: Subset,
) -> None:
    print("Episode-level split")
    print(f"  train_episodes = {len(train_episode_names)}")
    print(f"  val_episodes = {len(val_episode_names)}")
    print(f"  train_rows = {len(train_subset)}")
    print(f"  val_rows = {len(val_subset)}")
    print(f"  train_episode_names = {train_episode_names}")
    print(f"  val_episode_names = {val_episode_names}")


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    normalization_stats: DatasetStatistics,
    normalize_action: bool,
) -> tuple[float, torch.Tensor, float, torch.Tensor]:
    model.eval()
    total_normalized_loss = 0.0
    total_raw_loss = 0.0
    normalized_joint_sum = torch.zeros(6, dtype=torch.float32)
    raw_joint_sum = torch.zeros(6, dtype=torch.float32)
    batch_count = 0

    with torch.no_grad():
        for batch in dataloader:
            predicted_action_chunk = model(
                batch["wrist_image"],
                batch["base_image"],
                batch["robot_state"],
                batch["instruction_id"],
            )
            target_action_chunk = batch["action_chunk"]
            target_action_chunk_raw = batch["action_chunk_raw"]

            normalized_loss, normalized_joint = compute_loss_breakdown(
                predicted_action_chunk,
                target_action_chunk,
            )

            if normalize_action:
                predicted_action_chunk_raw = normalization_stats.denormalize_action(predicted_action_chunk)
            else:
                predicted_action_chunk_raw = predicted_action_chunk
            raw_loss, raw_joint = compute_loss_breakdown(
                predicted_action_chunk_raw,
                target_action_chunk_raw,
            )

            total_normalized_loss += normalized_loss.item()
            total_raw_loss += raw_loss.item()
            normalized_joint_sum += normalized_joint.cpu()
            raw_joint_sum += raw_joint.cpu()
            batch_count += 1

    if batch_count == 0:
        raise ValueError("Validation dataloader is empty.")

    return (
        total_normalized_loss / batch_count,
        normalized_joint_sum / batch_count,
        total_raw_loss / batch_count,
        raw_joint_sum / batch_count,
    )


def main() -> None:
    args = parse_args()
    excluded_episodes = set(args.exclude_episode)

    dataset = RealRobotEpisodeCollection(
        args.dataset_dir,
        action_chunk_size=ACTION_CHUNK_SIZE,
        skip_invalid_rows=args.skip_invalid_rows,
        skip_corrupted_episodes=args.skip_corrupted_episodes,
        excluded_episodes=excluded_episodes,
        require_base_image=True,
    )

    train_episode_indices, val_episode_indices = build_episode_splits(
        dataset=dataset,
        val_ratio=args.val_ratio,
        seed=args.seed,
        overfit_episodes=args.overfit_episodes,
    )
    train_episode_names = get_episode_names(dataset, train_episode_indices)
    val_episode_names = get_episode_names(dataset, val_episode_indices)

    normalization_stats = compute_dataset_statistics(
        dataset_dir=args.dataset_dir,
        require_base_image=True,
        excluded_episodes=excluded_episodes,
        included_episodes=set(train_episode_names),
    )

    dataset = RealRobotEpisodeCollection(
        args.dataset_dir,
        action_chunk_size=ACTION_CHUNK_SIZE,
        skip_invalid_rows=args.skip_invalid_rows,
        skip_corrupted_episodes=args.skip_corrupted_episodes,
        excluded_episodes=excluded_episodes,
        require_base_image=True,
        normalization_stats=normalization_stats if (args.normalize_state or args.normalize_action) else None,
        normalize_state=args.normalize_state,
        normalize_action=args.normalize_action,
    )

    train_subset = build_row_subset_from_episode_indices(dataset, train_episode_indices)
    val_subset = build_row_subset_from_episode_indices(dataset, val_episode_indices)
    train_dataloader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)

    run_suffix = "full_data" if args.overfit_episodes == 0 else f"overfit_{args.overfit_episodes}_episodes"
    run_name = f"{BASELINE_NAME}_{run_suffix}"
    loss_csv_path = PROJECT_ROOT / "data" / f"{run_name}_training_metrics.csv"
    model_path = PROJECT_ROOT / "data" / f"{run_name}_model.pth"
    loss_csv_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Baseline configuration = {BASELINE_NAME}")
    print(f"Run name = {run_name}")
    print(
        "Defaults:"
        f" skip_invalid_rows={args.skip_invalid_rows}"
        f" skip_corrupted_episodes={args.skip_corrupted_episodes}"
        f" normalize_state={args.normalize_state}"
        f" normalize_action={args.normalize_action}"
    )
    print_split_summary(train_episode_names, val_episode_names, train_subset, val_subset)
    print_statistics(normalization_stats)

    model = ActionChunkingTransformer(
        state_dim=6,
        action_dim=6,
        num_instructions=len(dataset.instruction_to_index),
        use_base_image=dataset.has_base_image,
        action_chunk_size=ACTION_CHUNK_SIZE,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    epoch_rows: list[list[float]] = []

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(train_dataloader, start=1):
            wrist_image = batch["wrist_image"]
            base_image = batch["base_image"]
            robot_state = batch["robot_state"]
            instruction_id = batch["instruction_id"]
            target_action_chunk = batch["action_chunk"]
            target_action_chunk_raw = batch["action_chunk_raw"]

            predicted_action_chunk = model(wrist_image, base_image, robot_state, instruction_id)
            if tuple(predicted_action_chunk.shape) != tuple(target_action_chunk.shape):
                raise ValueError(
                    "Model output shape does not match target shape: "
                    f"{tuple(predicted_action_chunk.shape)} vs {tuple(target_action_chunk.shape)}"
                )

            train_loss, train_joint_loss = compute_loss_breakdown(
                predicted_action_chunk,
                target_action_chunk,
            )

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            epoch_loss += train_loss.item()
            if step % args.log_every == 0 or step == 1 or step == len(train_dataloader):
                print(
                    f"epoch {epoch + 1}/{args.num_epochs} "
                    f"train_step {step}/{len(train_dataloader)} "
                    f"loss={train_loss.item():.4f}"
                )

            if step <= args.debug_batches:
                if args.normalize_action:
                    predicted_action_chunk_raw = normalization_stats.denormalize_action(predicted_action_chunk.detach())
                else:
                    predicted_action_chunk_raw = predicted_action_chunk.detach()
                _, train_joint_loss_raw = compute_loss_breakdown(
                    predicted_action_chunk_raw,
                    target_action_chunk_raw,
                )
                print(
                    "  shapes:"
                    f" output={tuple(predicted_action_chunk.shape)}"
                    f" target={tuple(target_action_chunk.shape)}"
                )
                print(
                    "  batch action range:"
                    f" target_min={target_action_chunk.min().item():.4f}"
                    f" target_max={target_action_chunk.max().item():.4f}"
                    f" raw_target_min={target_action_chunk_raw.min().item():.4f}"
                    f" raw_target_max={target_action_chunk_raw.max().item():.4f}"
                )
                print(
                    "  model output range:"
                    f" pred_min={predicted_action_chunk.min().item():.4f}"
                    f" pred_max={predicted_action_chunk.max().item():.4f}"
                    f" raw_pred_min={predicted_action_chunk_raw.min().item():.4f}"
                    f" raw_pred_max={predicted_action_chunk_raw.max().item():.4f}"
                )
                print(f"  train_loss_per_joint = {format_tensor_list(train_joint_loss.detach())}")
                print(f"  train_raw_loss_per_joint = {format_tensor_list(train_joint_loss_raw.detach())}")

        train_average_loss = epoch_loss / len(train_dataloader)
        (
            val_normalized_loss,
            val_normalized_joint_loss,
            val_raw_loss,
            val_raw_joint_loss,
        ) = evaluate_model(
            model=model,
            dataloader=val_dataloader,
            normalization_stats=normalization_stats,
            normalize_action=args.normalize_action,
        )

        print(
            f"epoch {epoch + 1} summary:"
            f" train_loss={train_average_loss:.4f}"
            f" val_loss={val_normalized_loss:.4f}"
            f" val_raw_loss={val_raw_loss:.4f}"
        )
        print(f"  val_loss_per_joint = {format_tensor_list(val_normalized_joint_loss)}")
        print(f"  val_raw_loss_per_joint = {format_tensor_list(val_raw_joint_loss)}")

        epoch_rows.append(
            [
                epoch + 1,
                train_average_loss,
                val_normalized_loss,
                val_raw_loss,
                *val_normalized_joint_loss.tolist(),
                *val_raw_joint_loss.tolist(),
            ]
        )

    with loss_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "val_loss",
                "val_raw_loss",
                *[f"val_loss_joint_{index}" for index in range(6)],
                *[f"val_raw_loss_joint_{index}" for index in range(6)],
            ]
        )
        writer.writerows(epoch_rows)

    torch.save(model.state_dict(), model_path)
    print(f"saved baseline model weights to {model_path}")
    print(f"saved baseline metrics to {loss_csv_path}")


if __name__ == "__main__":
    main()
