from abc import ABC, abstractmethod
import numpy as np

class ScreenCapture(ABC):
    @abstractmethod
    def capture(self) -> np.ndarray:
        """
        Captura la imagen de la aplicación a seguir.
        Retorna un arreglo NumPy que representa la imagen.
        """
        pass
