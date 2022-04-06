from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from numpy import ndarray


@dataclass
class Loss(ABC):
    """Abstract class which every loss type class would inherit its methods from"""

    def calculate(self, output, y) -> ndarray:
        """data loss calculation method"""
        sample_losses = self.loss_forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

    @abstractmethod
    def loss_forward(self, output, y) -> ndarray:
        """A method which returns a ndarray of losses for each sample"""
        pass
