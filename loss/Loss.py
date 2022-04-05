from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from numpy import ndarray


@dataclass
class Loss(ABC):
    """Parent abstract class which contains methods that are used in every loss calculating child class"""

    def calculate(self, output, y) -> ndarray:
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

    @abstractmethod
    def forward(self, output, y) -> ndarray:
        pass
