import cv2


BACKENDS = [
    ("dshow", cv2.CAP_DSHOW),
    ("msmf", cv2.CAP_MSMF),
    ("default", cv2.CAP_ANY),
]


def try_open_camera(index: int) -> tuple[str, cv2.VideoCapture] | None:
    for backend_name, backend_flag in BACKENDS:
        cap = cv2.VideoCapture(index, backend_flag)
        if not cap.isOpened():
            cap.release()
            continue

        ok = False
        for _ in range(5):
            ok, _ = cap.read()
            if ok:
                break

        if ok:
            return backend_name, cap

        cap.release()

    return None


def main() -> None:
    max_index = 10
    working_cameras: list[tuple[int, str]] = []

    for index in range(max_index):
        result = try_open_camera(index)

        if result is None:
            print(f"Camera index {index}: not working")
            continue

        backend_name, cap = result
        print(f"Camera index {index}: working via {backend_name}")
        working_cameras.append((index, backend_name))
        cap.release()

    print()
    print("Working cameras:", working_cameras)


if __name__ == "__main__":
    main()
