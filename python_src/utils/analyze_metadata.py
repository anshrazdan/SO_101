from pathlib import Path
import csv
import statistics


def main() -> None:
    metadata_path = Path("data/pilot_dataset/episode_1/metadata.csv")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with metadata_path.open("r", newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    total_rows = len(rows)
    state_columns = [f"state_{i}" for i in range(6)]
    action_columns = [f"action_{i}" for i in range(6)]

    zero_action_rows = 0
    repeated_states = 0
    seen_states: set[tuple[float, ...]] = set()

    state_values = {column: [] for column in state_columns}
    action_values = {column: [] for column in action_columns}

    for row in rows:
        state = tuple(float(row[column]) for column in state_columns)
        action = [float(row[column]) for column in action_columns]

        if all(value == 0.0 for value in action):
            zero_action_rows += 1

        if state in seen_states:
            repeated_states += 1
        else:
            seen_states.add(state)

        for column, value in zip(state_columns, state, strict=True):
            state_values[column].append(value)

        for column, value in zip(action_columns, action, strict=True):
            action_values[column].append(value)

    zero_action_percentage = (zero_action_rows / total_rows * 100.0) if total_rows > 0 else 0.0

    print(f"Total rows: {total_rows}")
    print(f"Zero-action rows: {zero_action_rows}")
    print(f"Zero-action percentage: {zero_action_percentage:.2f}%")
    print(f"Repeated robot states: {repeated_states}")
    print()

    print("State column statistics:")
    for column in state_columns:
        values = state_values[column]
        print(
            f"{column}: min={min(values):.4f}, max={max(values):.4f}, "
            f"mean={statistics.mean(values):.4f}, stdev={statistics.pstdev(values):.4f}"
        )

    print()
    print("Action column statistics:")
    for column in action_columns:
        values = action_values[column]
        print(
            f"{column}: min={min(values):.4f}, max={max(values):.4f}, "
            f"mean={statistics.mean(values):.4f}, stdev={statistics.pstdev(values):.4f}"
        )


if __name__ == "__main__":
    main()
