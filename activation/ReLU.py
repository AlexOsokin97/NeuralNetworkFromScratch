import numpy as np
from numpy import ndarray
from dataclasses import dataclass, field


@dataclass
class Relu:
    """Layer activation function which returns the largest value: 0 or i"""

    __output: ndarray = field(init=False)

    def activate(self, inputs: ndarray) -> None:
        self.output = np.maximum(0, inputs)

    @property
    def output(self) -> ndarray:
        return self.__output

    @output.setter
    def output(self, new_output) -> None:
        self.__output = new_output
