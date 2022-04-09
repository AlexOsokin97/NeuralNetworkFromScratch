import numpy as np
from nnfs.datasets import spiral_data
from layer.Layer_Dense import Layer_Dense
from activation.Softmax import Softmax
from activation.ReLU import Relu
from loss.CategoricalCrossEntropy import CategoricalCrossEntropy
from activation_loss_combo.SoftmaxCategoricalCrossEntropy import SoftmaxCategoricalCrossEntropy
from accuracy.Accuracy import Accuracy


def main() -> None:

    X, y = spiral_data(samples=500, classes=4)

    layer1 = Layer_Dense(2, 8, False)
    layer2 = Layer_Dense(8, 16, False)
    layer3 = Layer_Dense(16, 8, False)
    layer4 = Layer_Dense(8, 4, False)

    relu = Relu()
    softmax = Softmax()
    loss = CategoricalCrossEntropy()
    smcc = SoftmaxCategoricalCrossEntropy()

    #layer1.forward(X)
    #relu.activate(layer1.outputs)

    #layer2.forward(relu.output)
    #relu.activate(layer2.outputs)

    #layer3.forward(relu.output)
    #relu.activate(layer3.outputs)

    #layer4.forward(relu.output)

    softmax_outputs = np.array([[0.7, 0.1, 0.2],
                                [0.1, 0.5, 0.4],
                                [0.02, 0.9, 0.08]])

    class_targets = np.array([0, 1, 1])

    smcc.backward(softmax_outputs, class_targets)
    print(smcc.dinputs)

    softmax.outputs = softmax_outputs
    loss.backward(softmax_outputs, class_targets)
    softmax.backward(loss.dinputs)
    print(softmax.dinputs)

if __name__ == "__main__":
    main()
