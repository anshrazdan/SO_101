import time

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


def main() -> None:
    robot = SO101Follower(SO101FollowerConfig(port="COM6", id="follower"))
    robot.connect(calibrate=False)

    try:
        while True:
            observation = robot.get_observation()
            joint_state = {
                key: value
                for key, value in observation.items()
                if key.endswith(".pos")
            }
            print(joint_state)
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
