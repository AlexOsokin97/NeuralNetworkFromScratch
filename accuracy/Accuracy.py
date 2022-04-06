from dataclasses import dataclass, field
import numpy as np
from numpy import ndarray


@dataclass
class Accuracy:

    __accuracy: ndarray = field(init=False)

    def calculate(self, y_hat: ndarray, y_true: ndarray) -> None:

        predictions = np.argmax(y_hat, axis=1)

        if len(y_true.shape) == 1:
            self.accuracy = np.mean(predictions == y_true)

        elif len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
            self.accuracy = np.mean(predictions == y_true)

    @property
    def accuracy(self) -> ndarray:
        return self.__accuracy

    @accuracy.setter
    def accuracy(self, new_accuracy: ndarray) -> None:
        self.__accuracy = new_accuracy
