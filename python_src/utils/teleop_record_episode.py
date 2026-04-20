"""Record a teleoperated real-robot episode as wrist images and CSV metadata."""

import argparse
from pathlib import Path
import csv
import shutil
import time

import cv2

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader.so_leader import SO101Leader
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig


BACKENDS = [
    ("dshow", cv2.CAP_DSHOW),
    ("msmf", cv2.CAP_MSMF),
    ("default", cv2.CAP_ANY),
]


def rotate_wrist_frame(frame):
    return cv2.rotate(frame, cv2.ROTATE_180)


def observation_to_state(observation: dict[str, float]) -> list[float]:
    return [
        observation["shoulder_pan.pos"],
        observation["shoulder_lift.pos"],
        observation["elbow_flex.pos"],
        observation["wrist_flex.pos"],
        observation["wrist_roll.pos"],
        observation["gripper.pos"],
    ]


def open_camera(index: int) -> tuple[cv2.VideoCapture, str]:
    for backend_name, backend_flag in BACKENDS:
        cap = cv2.VideoCapture(index, backend_flag)
        if not cap.isOpened():
            cap.release()
            continue

        for _ in range(5):
            ok, _ = cap.read()
            if ok:
                return cap, backend_name

        cap.release()

    raise RuntimeError(f"Could not open camera index {index}.")


def open_camera_pair(
    wrist_index: int,
    base_index: int | None,
) -> tuple[cv2.VideoCapture, str, cv2.VideoCapture | None, str | None]:
    wrist_cap, wrist_backend = open_camera(wrist_index)

    if base_index is None:
        return wrist_cap, wrist_backend, None, None

    try:
        base_cap, base_backend = open_camera(base_index)
        return wrist_cap, wrist_backend, base_cap, base_backend
    except RuntimeError:
        wrist_cap.release()
        print("Opening wrist first failed for the second camera; retrying with base first.")

    base_cap, base_backend = open_camera(base_index)
    wrist_cap, wrist_backend = open_camera(wrist_index)
    return wrist_cap, wrist_backend, base_cap, base_backend


def next_episode_dir(dataset_dir: Path) -> Path:
    dataset_dir.mkdir(parents=True, exist_ok=True)

    episode_index = 1
    while True:
        episode_dir = dataset_dir / f"episode_{episode_index}"
        if not episode_dir.exists():
            return episode_dir
        episode_index += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wrist-camera-index", type=int, default=1)
    parser.add_argument("--base-camera-index", type=int, default=None)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/pilot_dataset"))
    parser.add_argument("--episode-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.episode_dir or next_episode_dir(args.dataset_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.csv"
    episode_name = output_dir.name
    print()
    print(f"Starting {episode_name}")
    print(f"Recording to {output_dir}")
    print("Press q in a camera window to save and stop this episode.")

    teleop = SO101Leader(SOLeaderTeleopConfig(port="COM5", id="leader"))
    robot = SO101Follower(SO101FollowerConfig(port="COM6", id="follower"))
    wrist_cap = None
    base_cap = None
    teleop_connected = False
    robot_connected = False

    try:
        wrist_cap, wrist_backend, base_cap, base_backend = open_camera_pair(
            args.wrist_camera_index,
            args.base_camera_index,
        )
        print(f"Wrist camera {args.wrist_camera_index} opened via {wrist_backend}")

        if base_cap is not None:
            print(f"Base camera {args.base_camera_index} opened via {base_backend}")

        teleop.connect(calibrate=False)
        teleop_connected = True
        robot.connect(calibrate=False)
        robot_connected = True
    except Exception:
        if wrist_cap is not None:
            wrist_cap.release()
        if base_cap is not None:
            base_cap.release()
        cv2.destroyAllWindows()
        if teleop_connected:
            teleop.disconnect()
        if robot_connected:
            robot.disconnect()
        if output_dir.exists() and not any(output_dir.iterdir()):
            output_dir.rmdir()
        raise

    metadata_file = metadata_path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(metadata_file)
    header = ["frame"]
    if base_cap is not None:
        header += ["base_frame"]
    header += [
        "timestamp",
        "state_0",
        "state_1",
        "state_2",
        "state_3",
        "state_4",
        "state_5",
        "action_0",
        "action_1",
        "action_2",
        "action_3",
        "action_4",
        "action_5",
        "instruction",
    ]
    writer.writerow(header)

    frame_index = 0
    rows_written = 0
    instruction = "hang the towel"
    pending_row: dict | None = None
    fps = 5

    try:
        while True:
            loop_start = time.perf_counter()

            current_observation = robot.get_observation()
            current_state = observation_to_state(current_observation)

            wrist_ok, wrist_frame = wrist_cap.read()
            if not wrist_ok:
                break

            wrist_frame = rotate_wrist_frame(wrist_frame)
            cv2.imshow("Wrist Camera", wrist_frame)
            frame_name = f"frame_{frame_index:03d}.png"
            wrist_image_path = output_dir / frame_name
            cv2.imwrite(str(wrist_image_path), wrist_frame)

            base_frame_name: str | None = None
            if base_cap is not None:
                base_ok, base_frame = base_cap.read()
                if not base_ok:
                    break

                cv2.imshow("Base Camera", base_frame)
                base_frame_name = f"base_frame_{frame_index:03d}.png"
                base_image_path = output_dir / base_frame_name
                cv2.imwrite(str(base_image_path), base_frame)

            timestamp = time.time()

            leader_action = teleop.get_action()
            robot.send_action(leader_action)

            current_row = {
                "frame": frame_name,
                "base_frame": base_frame_name,
                "timestamp": timestamp,
                "state": current_state,
            }

            if pending_row is not None:
                action = [
                    next_value - current_value
                    for current_value, next_value in zip(pending_row["state"], current_state, strict=True)
                ]
                row = [pending_row["frame"]]
                if base_cap is not None:
                    row += [pending_row["base_frame"]]
                row += [pending_row["timestamp"]]
                row += pending_row["state"]
                row += action
                row += [instruction]
                writer.writerow(row)
                metadata_file.flush()
                rows_written += 1

            pending_row = current_row
            frame_index += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            elapsed = time.perf_counter() - loop_start
            time.sleep(max(1 / fps - elapsed, 0.0))
    finally:
        if pending_row is not None:
            row = [pending_row["frame"]]
            if base_cap is not None:
                row += [pending_row["base_frame"]]
            row += [pending_row["timestamp"]]
            row += pending_row["state"]
            row += [0.0] * 6
            row += [instruction]
            writer.writerow(row)
            metadata_file.flush()
            rows_written += 1

        metadata_file.close()
        wrist_cap.release()
        if base_cap is not None:
            base_cap.release()
        cv2.destroyAllWindows()
        if teleop_connected:
            teleop.disconnect()
        if robot_connected:
            robot.disconnect()
        print()
        print(f"Finished recording {episode_name}.")
        print(f"  folder: {output_dir}")
        print(f"  metadata: {metadata_path}")
        print(f"  frames captured: {frame_index}")
        print(f"  metadata rows written: {rows_written}")
        print()
        keep_episode = input("Keep this episode? Type y to save, or n to delete and try again: ").strip().lower()
        if keep_episode in {"n", "no"}:
            shutil.rmtree(output_dir)
            print(f"Deleted {episode_name}. Rerun the same command to try this episode again.")
        else:
            print(f"Saved {episode_name}. Rerun the same command to record the next episode.")


if __name__ == "__main__":
    main()
