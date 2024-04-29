import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMClassifier

from .base_model import BaseModel


class LightGBM(BaseModel):
    _model: LGBMClassifier

    def __init__(self, parameters=None):
        super().__init__(parameters)

        self._model = LGBMClassifier(**self.parameters, verbose=0)

    def fit(self, train_X: np.ndarray, train_Y: np.array):
        assert len(train_Y.shape) == 1, "train_Y should be 1D"

        self._model.fit(train_X, train_Y)

    def predict(self, X: np.ndarray) -> np.array:
        return self._model.predict(X)
