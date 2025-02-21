# internal/domain/simulator/drive_use_case.py
import cv2

class DriveUseCase:
    def __init__(self, screen_capture, keyboard_controller):
        self.screen_capture = screen_capture
        self.keyboard = keyboard_controller

    def drive(self):
        frame = self.screen_capture.capture()
        cv2.imshow("Captura de Ventana", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
