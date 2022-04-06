import numpy as np
from nnfs.datasets import spiral_data
from layer.Layer_Dense import Layer_Dense
from activation.ReLU import Relu
from activation.Softmax import Softmax
from loss.CategoricalCrossEntropy import CategoricalCrossEntropy
from accuracy.Accuracy import Accuracy


def main() -> None:

    X, y = spiral_data(samples=500, classes=3)

    layer1 = Layer_Dense(2, 8, True)
    layer2 = Layer_Dense(8, 3, False)
    relu = Relu()
    sm = Softmax()
    loss_func = CategoricalCrossEntropy()
    accuracy = Accuracy()

    relu.activate(layer1.forward(X))
    sm.activate(layer2.forward(relu.output))

    loss = loss_func.calculate(sm.probabilities, y)
    print(loss)


if __name__ == "__main__":
    main()
