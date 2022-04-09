import numpy as np
from numpy import ndarray
from dataclasses import dataclass, field


@dataclass
class Softmax:
    """Output layer activation function which is implemented on the output layer of a Neural Network
        which returns a ndarray of probabilities [S = e^(Z) / Sum(e^(Z))]"""

    __outputs: ndarray = field(init=False)
    __dinputs: ndarray = field(init=False)

    def activate(self, inputs: ndarray) -> None:
        # get non-normalized exponential values
        # subtract the max input value from each input value sample-wise
        # in order to prevent explosion of values (overflow error)
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # lastly normalizing each output by the sum of all outputs for each sample (row-wise)
        self.outputs = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def backward(self, dvalues: ndarray):
        # uninitialized array of sample gradients
        self.dinputs = np.empty_like(dvalues)

        # enumerate outputs (probabilities) and gradients respectively
        for index, (single_output, single_dvalues) in enumerate(zip(self.outputs, dvalues)):
            # flatten output array
            single_output = single_output.reshape(-1, 1)
            # create jacobian matrix which is the derivative product of the softmax activation function
            # dsoftmax = S(i,j) * KroneckerDelta(i,j) - S(i,j) * S(i,k)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

    @property
    def outputs(self) -> ndarray:
        return self.__outputs

    @outputs.setter
    def outputs(self, new_outputs) -> None:
        self.__outputs = new_outputs

    @property
    def dinputs(self) -> ndarray:
        return self.__dinputs

    @dinputs.setter
    def dinputs(self, derivatives) -> None:
        self.__dinputs = derivatives

