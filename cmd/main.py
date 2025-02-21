# cmd/main.py
from internal.domain.simulator.drive_use_case import DriveUseCase
from internal.infrastructure.window_capture import QuartzWindowCapture
from internal.infrastructure.keyboard_controller import KeyboardController


def main():
    screen_capture = QuartzWindowCapture("Euro truck")
    keyboard_controller = KeyboardController()
    drive_use_case = DriveUseCase(screen_capture, keyboard_controller)
    drive_use_case.drive()


if __name__ == '__main__':
    main()
