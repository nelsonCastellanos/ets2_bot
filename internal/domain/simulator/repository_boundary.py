from abc import ABC, abstractmethod
import numpy as np

class ScreenCapture(ABC):
    @abstractmethod
    def capture(self) -> np.ndarray:
        """
        Captura la imagen de la aplicación (ventana) que se desea seguir.
        Retorna un arreglo de NumPy que representa la imagen.
        """
        pass
