import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from .base_model import BaseModel


class DecisionTree(BaseModel):
    _model: DecisionTreeClassifier

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self._model = DecisionTreeClassifier(**self.parameters)

    def fit(self, train_X: np.ndarray, train_Y: np.array):
        assert len(train_Y.shape) == 1, "train_Y should be 1D"

        self._model.fit(train_X, train_Y)

    def predict(self, X: np.ndarray) -> np.array:
        return self._model.predict(X)
