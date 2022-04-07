import numpy as np
from numpy import ndarray
from dataclasses import dataclass, field


@dataclass
class Relu:
    """Layer activation function which returns the largest value: 0 or i"""

    __inputs: ndarray = field(init=False)
    __output: ndarray = field(init=False)
    __dinputs: ndarray = field(init=False)

    def activate(self, inputs: ndarray) -> None:
        """activation method of the ReLU"""
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues: ndarray) -> None:
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

    @property
    def inputs(self) -> ndarray:
        return self.__output

    @inputs.setter
    def inputs(self, new_inputs) -> None:
        self.__inputs = new_inputs

    @property
    def output(self) -> ndarray:
        return self.__output

    @output.setter
    def output(self, new_output) -> None:
        self.__output = new_output

    @property
    def dinputs(self) -> ndarray:
        return self.__dinputs

    @dinputs.setter
    def dinputs(self, values: ndarray) -> None:
        self.__dinputs = values
