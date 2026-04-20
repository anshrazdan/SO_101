import argparse

import cv2


BACKENDS = [
    ("dshow", cv2.CAP_DSHOW),
    ("msmf", cv2.CAP_MSMF),
    ("default", cv2.CAP_ANY),
]


def open_camera(index: int) -> tuple[cv2.VideoCapture, str] | None:
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

    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", nargs="+", type=int, default=[0, 1, 2])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cameras: list[tuple[int, str, cv2.VideoCapture]] = []

    for index in args.indices:
        result = open_camera(index)
        if result is None:
            print(f"Camera {index}: could not open")
            continue

        cap, backend_name = result
        cameras.append((index, backend_name, cap))
        print(f"Camera {index}: opened via {backend_name}")

    if not cameras:
        raise RuntimeError("No cameras opened.")

    print("Press q to quit.")

    try:
        while True:
            for index, backend_name, cap in cameras:
                ok, frame = cap.read()
                if not ok:
                    continue

                label = f"index {index} ({backend_name})"
                cv2.putText(
                    frame,
                    label,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(f"Camera {index}", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        for _, _, cap in cameras:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
