import numpy as np
from numpy import ndarray
from dataclasses import dataclass, field


@dataclass
class Softmax:
    """Output layer activation function which is implemented on the output layer of a Neural Network
        which returns a ndarray of probabilities [S = e^(Z) / Sum(e^(Z))]"""

    __probabilities: ndarray = field(init=False)
    __dinputs: ndarray = field(init=False)

    def activate(self, inputs: ndarray) -> None:
        # get non-normalized exponential values
        # subtract the max input value from each input value sample-wise
        # in order to prevent explosion of values (overflow error)
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # lastly we normalize each output by the sum of all outputs for each sample (row-wise)
        self.probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def backward(self, dvalues: ndarray):
        # uninitialized array of sample gradients
        self.dinputs = np.empty_like(dvalues)

        # enumerate outputs (probabilities) and gradients respectively
        for index, (single_output, single_dvalues) in enumerate(zip(self.probabilities, dvalues)):
            # create jacobian matrix which is the derivative product of the softmax activation function
            # dsoftmax = S(i,j) * KroneckerDelta(i,j) - S(i,j) * S(i,k)
            jacobian_matrix = np.diagflat(single_output.reshape(-1, 1)) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

    @property
    def probabilities(self) -> ndarray:
        return self.__probabilities

    @probabilities.setter
    def probabilities(self, new_probabilities) -> None:
        self.__probabilities = new_probabilities

    @property
    def dinputs(self) -> ndarray:
        return self.__dinputs

    @dinputs.setter
    def dinputs(self, derivatives) -> None:
        self.__dinputs = derivatives

