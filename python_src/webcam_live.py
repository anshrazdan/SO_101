import argparse
from pathlib import Path

import cv2


BACKENDS = [
    ("dshow", cv2.CAP_DSHOW),
    ("msmf", cv2.CAP_MSMF),
    ("default", cv2.CAP_ANY),
]


def rotate_wrist_frame(frame):
    return cv2.rotate(frame, cv2.ROTATE_180)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wrist-camera-index", type=int, default=1)
    parser.add_argument("--base-camera-index", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_dir = Path("data/test_images")
    save_dir.mkdir(parents=True, exist_ok=True)

    wrist_cap, wrist_backend, base_cap, base_backend = open_camera_pair(
        args.wrist_camera_index,
        args.base_camera_index,
    )
    print(f"Wrist camera {args.wrist_camera_index} opened via {wrist_backend}")

    if base_cap is not None:
        print(f"Base camera {args.base_camera_index} opened via {base_backend}")

    img_count = 0

    try:
        while True:
            wrist_ok, wrist_frame = wrist_cap.read()
            if not wrist_ok:
                print("Failed to grab wrist frame")
                break

            wrist_frame = rotate_wrist_frame(wrist_frame)
            cv2.imshow("Wrist Camera", wrist_frame)

            base_frame = None
            if base_cap is not None:
                base_ok, base_frame = base_cap.read()
                if not base_ok:
                    print("Failed to grab base frame")
                    break
                cv2.imshow("Base Camera", base_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                wrist_path = save_dir / f"wrist_{img_count:03d}.png"
                cv2.imwrite(str(wrist_path), wrist_frame)
                print(f"Saved {wrist_path}")

                if base_frame is not None:
                    base_path = save_dir / f"base_{img_count:03d}.png"
                    cv2.imwrite(str(base_path), base_frame)
                    print(f"Saved {base_path}")

                img_count += 1
            elif key == ord("q"):
                break
    finally:
        wrist_cap.release()
        if base_cap is not None:
            base_cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
