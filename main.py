import numpy as np
from nnfs.datasets import spiral_data, vertical_data
from numpy import ndarray
from layer.Layer_Dense import Layer_Dense
from activation.ReLU import Relu
from activation.Softmax import Softmax
from loss.CategoricalCrossEntropy import CategoricalCrossEntropy
from accuracy.Accuracy import Accuracy


def main() -> None:

    softmax = np.array([[0.6, -0.023, 0.4, 0.6, -0.1]])
    print(np.empty_like(softmax))


if __name__ == "__main__":
    main()
