from abc import ABC, abstractmethod

class ModelTrainer(ABC):
    @abstractmethod
    def train(self) -> None:
        """
        Ejecuta el proceso de entrenamiento del modelo.
        """
        pass
