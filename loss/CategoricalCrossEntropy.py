from dataclasses import dataclass, field
import numpy as np
from numpy import ndarray

from loss.Loss import Loss


@dataclass
class CategoricalCrossEntropy(Loss):

    __loss: float = field(init=False)

    def forward(self, y_hat, y_true) -> ndarray:
        # How many samples there are (rows)
        samples = len(y_hat)
        # Clipping the predicted values between 1e-7 (close to zero) in order
        # to prevent a 0 value (which will result in inf error)
        # and subtracting 1-1e-7 in order to not create biased values
        y_hat_clipped = np.clip(y_hat, 1e-7, 1-1e-7)
        correct_confidences = np.array([])

        # Check if ture values are scaler values [1, 0]
        if len(y_true.shape) == 1:
            correct_confidences = y_hat_clipped[range(samples), y_true]

        # Check if true values are one hot encoded [[1,0], [0,1]]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_hat_clipped*y_true, axis=1)

        neg_log_likelihoods = -np.log(correct_confidences)
        return neg_log_likelihoods


    @property
    def loss(self):
        return self.__loss

    @loss.setter
    def loss(self, new_loss):
        self.__loss = new_loss
