import numpy as np
import cv2
import mss
import pygetwindow as gw
from internal.domain.simulator.repository_boundary import ScreenCapture

class WindowCapture(ScreenCapture):
    def __init__(self):
        self.window = None

    def select_window(self):
        windows = [w for w in gw.getAllTitles() if w.strip()]
        if not windows:
            print("No se encontraron ventanas abiertas.")
            return None

        print("Ventanas disponibles:")
        for idx, title in enumerate(windows):
            print(f"{idx}: {title}")

        idx = int(input("Ingrese el índice de la ventana a capturar: "))
        self.window = gw.getWindowsWithTitle(windows[idx])[0]
        print(f"Ventana seleccionada: {self.window.title}")

    def capture(self) -> np.ndarray:
        if self.window is None:
            self.select_window()
            if self.window is None:
                raise Exception("No se pudo seleccionar ninguna ventana.")

        # Obtenemos la posición y el tamaño de la ventana seleccionada
        region = {
            "top": self.window.top,
            "left": self.window.left,
            "width": self.window.width,
            "height": self.window.height
        }
        with mss.mss() as sct:
            frame = np.array(sct.grab(region))
            # Convertir de BGRA a BGR para OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame

    def capture_thumbnail(self, thumbnail_size=(320, 240)) -> np.ndarray:
        """
        Captura la imagen completa de la ventana y la redimensiona a un tamaño pequeño (thumbnail).
        """
        frame = self.capture()
        thumbnail = cv2.resize(frame, thumbnail_size)
        return thumbnail
