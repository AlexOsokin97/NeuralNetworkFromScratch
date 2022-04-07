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
    __weights: ndarray = field(init=False)
    __biases: ndarray = field(init=False)
    __inputs: ndarray = field(init=False)
    __dweights: ndarray = field(init=False)
    __dinputs: ndarray = field(init=False)
    __dbiases: ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.weights = self.__init_weights(self.weight_init_scaler)
        self.biases = self.__init_biases()

    def __init_weights(self, scaler: float | int) -> ndarray:
        """Returns a  Gaussian distribution ndarray of weights of
         shape (n_inputs, n_neurons) with mean of 0 and variance of 1"""

        if self.__weights is None:
            weights = scaler * np.random.randn(self.num_inputs, self.neurons)
            return weights
        elif self.__weights is not None:
            return self.__weights

    def __init_biases(self) -> ndarray:
        """Returns a ndarray of biases of shape (1, n_neurons) where each bias equals 0 OR
        a ndarray of Gaussian distribution of biases of shape (1, n_neurons)"""

        if self.bias_zero and self.biases is None:
            biases = np.zeros((1, self.neurons))
            return biases

        elif not self.bias_zero and self.biases is None:
            biases = np.random.randn(1, self.neurons)
            return biases

        elif self.biases is not None:
            return self.biases

    def forward(self, data: ndarray) -> ndarray:
        """Returns a ndarray of a dot product between input data and
         weights plus the bias (input * weight + bias)"""
        self.inputs = data
        return np.dot(data, self.weights) + self.biases

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
