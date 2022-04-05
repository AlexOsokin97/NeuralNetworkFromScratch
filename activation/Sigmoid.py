import numpy as np
from numpy import ndarray
from dataclasses import dataclass, field


@dataclass
class Sigmoid:
    """Layer activation functions which uses the Sigmoid function"""

    __output: ndarray = field(init=False)

    def activate(self, inputs) -> None:
        self.output = 1/1 + np.power(np.e, -inputs)

    @property
    def output(self) -> ndarray:
        return self.__output

    @output.setter
    def output(self, new_output) -> None:
        self.__output = new_output
        