import numpy as np
from numpy import ndarray
from activation.Softmax import Softmax
from loss.CategoricalCrossEntropy import CategoricalCrossEntropy
from dataclasses import dataclass, field


@dataclass
class SoftmaxCategoricalCrossEntropy:

    __softmax: Softmax = field(init=False)
    __loss: CategoricalCrossEntropy = field(init=False)
    __dinputs: ndarray = field(init=False)

    def forward(self, inputs: ndarray, y_true: ndarray) -> None:
        """passes the inputs from previous layer into a softmax activation function and
        then calculates loss using the outputs of the softmax"""
        self.softmax = Softmax()
        self.loss = CategoricalCrossEntropy()

        self.softmax.activate(inputs)
        self.loss.calculate(self.softmax.outputs, y_true)

    def backward(self, dvalues: ndarray, y_true: ndarray) -> None:
        """calculates the derivatives of the loss function and softmax activation function using the chain rule"""
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()

        self.dinputs[range(samples), y_true] -= 1

        self.dinputs = self.dinputs / samples

    @property
    def softmax(self) -> Softmax:
        return self.__softmax

    @softmax.setter
    def softmax(self, obj: Softmax) -> None:
        self.__softmax = obj

    @property
    def loss(self) -> CategoricalCrossEntropy:
        return self.__loss

    @loss.setter
    def loss(self, obj: CategoricalCrossEntropy) -> None:
        self.__loss = obj

    @property
    def dinputs(self) -> ndarray:
        return self.__dinputs

    @dinputs.setter
    def dinputs(self, derivatives: ndarray) -> None:
        self.__dinputs = derivatives
