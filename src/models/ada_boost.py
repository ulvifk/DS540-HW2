import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier

from .base_model import BaseModel


class AdaBoost(BaseModel):
    _model: AdaBoostClassifier

    def __init__(self, parameters=None):
        super().__init__(parameters)

        if parameters is not None:
            self._model = AdaBoostClassifier(**self.parameters)
        else:
            self._model = AdaBoostClassifier(algorithm='SAMME')

    def fit(self, train_X: np.ndarray, train_Y: np.array):
        assert len(train_Y.shape) == 1, "train_Y should be 1D"

        self._model.fit(train_X, train_Y)

    def predict(self, X: np.ndarray) -> np.array:
        return self._model.predict(X)
