from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
from numpy import ndarray


@dataclass
class Loss(ABC):
    """Abstract class which every loss type class would inherit its methods from"""

    __neg_log_likelihoods: ndarray = field(init=False)
    __data_loss: ndarray = field(init=False)

    def calculate(self, output: ndarray, y: ndarray) -> None:
        """data loss calculation method"""
        self.neg_log_likelihoods = self.loss_forward(output, y)
        self.data_loss = np.mean(self.neg_log_likelihoods)

    @abstractmethod
    def loss_forward(self, output: ndarray, y: ndarray) -> ndarray:
        """A method which returns a ndarray of losses for each sample"""
        pass

    @property
    def neg_log_likelihoods(self) -> ndarray:
        return self.__neg_log_likelihoods

    @neg_log_likelihoods.setter
    def neg_log_likelihoods(self, new_likelihoods) -> None:
        self.__neg_log_likelihoods = new_likelihoods

    @property
    def data_loss(self) -> ndarray:
        return self.__data_loss

    @data_loss.setter
    def data_loss(self, new_loss) -> None:
        self.__data_loss = new_loss