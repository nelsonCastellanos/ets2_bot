import Quartz
import Quartz.CoreGraphics as CG
import numpy as np
import cv2
from internal.domain.simulator.repository_boundary import ScreenCapture


class QuartzWindowCapture(ScreenCapture):
    def __init__(self, window_title: str):
        self.window_title = window_title
        self.target_window_id = None
        self.target_bounds = None
        self._find_window()

    def _find_window(self):
        window_list = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID
        )

        for window in window_list:
            title = window.get('kCGWindowName', '')
            if self.window_title.lower() in title.lower():
                self.target_window_id = window['kCGWindowNumber']
                self.target_bounds = window.get('kCGWindowBounds', None)
                break

        if self.target_window_id is None or self.target_bounds is None:
            print(f"No se encontró ninguna ventana con título que contenga '{self.window_title}'.")

    def capture(self) -> np.ndarray:
        if self.target_window_id is None or self.target_bounds is None:
            self._find_window()
            if self.target_window_id is None or self.target_bounds is None:
                raise Exception("No se pudo encontrar la ventana deseada.")

        # Definir el rectángulo de captura basado en los bounds de la ventana
        x = self.target_bounds.get('X', 0)
        y = self.target_bounds.get('Y', 0)
        width = self.target_bounds.get('Width', 0)
        height = self.target_bounds.get('Height', 0)
        capture_rect = CG.CGRectMake(x, y, width, height)

        # Capturar la imagen de la ventana
        image_ref = CG.CGWindowListCreateImage(
            capture_rect,
            Quartz.kCGWindowListOptionIncludingWindow,
            self.target_window_id,
            Quartz.kCGWindowImageDefault
        )

        if image_ref is None:
            print("No se pudo capturar la imagen de la ventana.")
            return None

        # Obtener las dimensiones y datos de la imagen capturada
        img_width = CG.CGImageGetWidth(image_ref)
        img_height = CG.CGImageGetHeight(image_ref)
        bytes_per_row = CG.CGImageGetBytesPerRow(image_ref)
        data_provider = CG.CGImageGetDataProvider(image_ref)
        data = CG.CGDataProviderCopyData(data_provider)

        # Convertir a un array NumPy; el formato es usualmente BGRA.
        buffer = np.frombuffer(data, dtype=np.uint8)
        # Ajustar según bytes_per_row
        image = buffer.reshape((img_height, bytes_per_row // 4, 4))[:, :img_width, :]

        # Convertir de BGRA a BGR para OpenCV
        bgr_image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return bgr_image