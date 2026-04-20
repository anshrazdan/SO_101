from pathlib import Path
import csv


def is_zero_action(row: dict[str, str]) -> bool:
    return all(float(row[f"action_{i}"]) == 0.0 for i in range(6))


def state_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[f"state_{i}"] for i in range(6))


def main() -> None:
    input_path = Path("data/pilot_dataset/episode_1/metadata.csv")
    output_path = Path("data/pilot_dataset/episode_1/metadata_clean.csv")

    if not input_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {input_path}")

    with input_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if fieldnames is None:
        raise ValueError("Metadata CSV is missing a header row.")

    cleaned_rows: list[dict[str, str]] = []
    previous_zero_idle_state: tuple[str, ...] | None = None

    for row in rows:
        zero_action = is_zero_action(row)
        current_state = state_key(row)

        if zero_action and current_state == previous_zero_idle_state:
            continue

        cleaned_rows.append(row)

        if zero_action:
            previous_zero_idle_state = current_state
        else:
            previous_zero_idle_state = None

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned_rows)

    original_count = len(rows)
    cleaned_count = len(cleaned_rows)
    removed_count = original_count - cleaned_count

    print(f"original row count: {original_count}")
    print(f"cleaned row count: {cleaned_count}")
    print(f"rows removed: {removed_count}")


if __name__ == "__main__":
    main()
