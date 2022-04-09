import numpy as np
from numpy import ndarray
from dataclasses import dataclass, field


@dataclass
class Layer_Dense:
    """Fully connected layer class"""

    __num_inputs: int
    __num_neurons: int
    __bias_zero: bool
    __weight_init_scaler: int | float = 0.01
    __weights: ndarray = None
    __biases: ndarray = None
    __inputs: ndarray = field(init=False)
    __outputs: ndarray = field(init=False)
    __dweights: ndarray = field(init=False)
    __dinputs: ndarray = field(init=False)
    __dbiases: ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.__init_weights(self.weight_init_scaler)
        self.__init_biases()

    def __init_weights(self, scaler: float | int) -> None:
        """sets weights to a  Gaussian distribution ndarray of weights of
         shape (n_inputs, n_neurons) with mean of 0 and variance of 1"""

        if self.weights is None:
            self.weights = scaler * np.random.randn(self.num_inputs, self.neurons)

    def __init_biases(self) -> None:
        """sets the biases to a ndarray of biases of shape (1, n_neurons) where each bias equals 0 OR
        a ndarray of Gaussian distribution of biases of shape (1, n_neurons)"""

        if self.bias_zero and self.biases is None:
            self.biases = np.zeros((1, self.neurons))

        elif not self.bias_zero and self.biases is None:
            self.biases = np.random.randn(1, self.neurons)

    def forward(self, data: ndarray) -> None:
        """sets outputs to a ndarray of dot product between input data and
         weights plus the bias (input * weight + bias)"""
        self.inputs = data
        self.outputs = np.dot(data, self.weights) + self.biases

    def backward(self, dvalues: ndarray) -> None:
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dinputs = np.dot(self.weights.T, dvalues)
        self.biases = np.sum(dvalues, axis=0, keepdims=True)

    @property
    def num_inputs(self) -> int:
        return self.__num_inputs

    @property
    def neurons(self) -> int:
        return self.__num_neurons

    @property
    def bias_zero(self) -> bool:
        return self.__bias_zero

    @property
    def weight_init_scaler(self):
        return self.__weight_init_scaler

    @property
    def weights(self) -> ndarray:
        return self.__weights

    @weights.setter
    def weights(self, new_weights: ndarray) -> None:
        self.__weights = new_weights

    @property
    def biases(self) -> ndarray:
        return self.__biases

    @biases.setter
    def biases(self, new_biases: ndarray) -> None:
        self.__biases = new_biases

    @property
    def inputs(self) -> ndarray:
        return self.__inputs

    @inputs.setter
    def inputs(self, new_inputs: ndarray) -> None:
        self.__inputs = new_inputs

    @property
    def outputs(self) -> ndarray:
        return self.__outputs

    @outputs.setter
    def outputs(self, new_outputs: ndarray) -> None:
        self.__outputs = new_outputs

    @property
    def dweights(self) -> ndarray:
        return self.__dweights

    @property
    def dinputs(self) -> ndarray:
        return self.__dinputs

    @property
    def dbiases(self) -> ndarray:
        return self.__dbiases

    @dweights.setter
    def dweights(self, values: ndarray) -> None:
        self.__dweights = values

    @dinputs.setter
    def dinputs(self, values: ndarray) -> None:
        self.__dinputs = values

    @dbiases.setter
    def dbiases(self, values: ndarray) -> None:
        self.__dbiases = values
