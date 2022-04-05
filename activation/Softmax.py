import numpy as np
from numpy import ndarray
from dataclasses import dataclass, field


@dataclass
class Softmax:
    """Output layer activation function which is implemented on the output layer of a Neural Network
        which returns a ndarray of probabilities [S = e^(Z) / Sum(e^(Z))]"""

    __probabilities: ndarray = field(init=False)

    def activate(self, inputs: ndarray) -> None:
        # get non-normalized exponential values
        # subtract the max input value from each input value sample-wise
        # in order to prevent explosion of values (overflow error)
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # lastly we normalize each output by the sum of all outputs for each sample (row-wise)
        self.probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    @property
    def probabilities(self) -> ndarray:
        return self.__probabilities

    @probabilities.setter
    def probabilities(self, new_probabilities) -> None:
        self.__probabilities = new_probabilities
