import numpy as np
from nnfs.datasets import spiral_data, vertical_data
from numpy import ndarray
from layer.Layer_Dense import Layer_Dense
from activation.ReLU import Relu
from activation.Softmax import Softmax
from loss.CategoricalCrossEntropy import CategoricalCrossEntropy
from accuracy.Accuracy import Accuracy


def main() -> None:

    X, y = spiral_data(samples=500, classes=4)

    layer1 = Layer_Dense(2, 8, False)
    layer2 = Layer_Dense(8, 16, False)
    layer3 = Layer_Dense(16, 32, False)
    layer4 = Layer_Dense(32, 16, False)
    layer5 = Layer_Dense(16, 8, False)
    layer6 = Layer_Dense(8, 4, False)

    relu = Relu()
    softmax = Softmax()
    loss_func = CategoricalCrossEntropy()
    accuracy = Accuracy()

    relu.activate(layer1.forward(X))
    relu.activate(layer2.forward(relu.output))
    relu.activate(layer3.forward(relu.output))
    relu.activate(layer4.forward(relu.output))
    relu.activate(layer5.forward(relu.output))
    softmax.activate(layer6.forward(relu.output))

    loss = loss_func.calculate(softmax.probabilities, y)
    accuracy.calculate(softmax.probabilities, y)

    print(layer1.forward(X[: 1]))
    print(loss)


if __name__ == "__main__":
    main()
