from dataclasses import dataclass


@dataclass
class SGD:

    __learning_rate: float = 0.001

    def update(self, layer):

        layer.weights += self.learning_rate * layer.dweights
        layer.biases += self.learning_rate * layer.dbiases

    @property
    def learning_rate(self) -> float:
        return self.__learning_rate

    @learning_rate.setter
    def learning_rate(self, new_rate) -> None:
        self.__learning_rate = new_rate
