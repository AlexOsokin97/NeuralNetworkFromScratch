import numpy as np
from numpy import ndarray
from dataclasses import dataclass


@dataclass
class Layer_Dense:
    """Fully connected layer class"""

    __num_inputs: int
    __num_neurons: int
    __bias_zero: bool
    __weight_init_scaler: int | float = 0.01
    __weights: ndarray = None
    __biases: ndarray = None

    def __post_init__(self) -> None:
        self.__weights = self.__init_weights(0.01)
        self.__biases = self.__init_biases()

    def __init_weights(self, scaler: float | int) -> ndarray:
        """Returns a  Gaussian distribution ndarray of weights of
         shape (n_inputs, n_neurons) with mean of 0 and variance of 1"""

        if self.__weights is None:
            weights = scaler * np.random.randn(self.inputs, self.neurons)
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

        return np.dot(data, self.weights) + self.biases

    @property
    def inputs(self) -> int:
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
    def weights(self, new_weights) -> None:
        self.__weights = new_weights

    @property
    def biases(self) -> ndarray:
        return self.__biases

    @biases.setter
    def biases(self, new_biases) -> None:
        self.__biases = new_biases
