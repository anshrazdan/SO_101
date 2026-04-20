from pathlib import Path
import csv
import time

import cv2
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


def rotate_wrist_frame(frame):
    return cv2.rotate(frame, cv2.ROTATE_180)


def main() -> None:
    output_dir = Path("data/pilot_dataset/episode_1")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.csv"
    robot = SO101Follower(SO101FollowerConfig(port="COM6", id="follower"))

    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera index 1.")

    robot.connect(calibrate=False)

    frame_index = 0
    instruction = "hang the towel"
    pending_row: dict | None = None

    metadata_file = metadata_path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(metadata_file)
    header = ["frame", "timestamp"]
    header += [f"state_{i}" for i in range(6)]
    header += [f"action_{i}" for i in range(6)]
    header += ["instruction"]
    writer.writerow(header)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = rotate_wrist_frame(frame)
            cv2.imshow("Wrist Camera", frame)
            image_path = output_dir / f"frame_{frame_index:03d}.png"
            cv2.imwrite(str(image_path), frame)

            timestamp = time.time()
            observation = robot.get_observation()
            robot_state = [
                observation["shoulder_pan.pos"],
                observation["shoulder_lift.pos"],
                observation["elbow_flex.pos"],
                observation["wrist_flex.pos"],
                observation["wrist_roll.pos"],
                observation["gripper.pos"],
            ]

            current_row = {
                "frame": image_path.name,
                "timestamp": timestamp,
                "state": robot_state,
            }

            if pending_row is not None:
                action = [
                    next_value - current_value
                    for current_value, next_value in zip(pending_row["state"], robot_state, strict=True)
                ]
                row = [pending_row["frame"], pending_row["timestamp"]]
                row += pending_row["state"]
                row += action
                row += [instruction]
                writer.writerow(row)
                metadata_file.flush()

            pending_row = current_row

            frame_index += 1

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
    finally:
        if pending_row is not None:
            final_action = [0.0] * 6
            row = [pending_row["frame"], pending_row["timestamp"]]
            row += pending_row["state"]
            row += final_action
            row += [instruction]
            writer.writerow(row)
            metadata_file.flush()
        metadata_file.close()
        robot.disconnect()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
