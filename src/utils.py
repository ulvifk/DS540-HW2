from sklearn.preprocessing import StandardScaler, MinMaxScaler

from .models import BaseModel
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
from imblearn.over_sampling import SMOTE


def calculate_accuracy_score(model: BaseModel, X_test: np.ndarray, y_test: np.array) -> float:
    assert len(y_test.shape) == 1, "y_test should be 1D"

    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)


def calculate_auc_score(model: BaseModel, X_test: np.ndarray, y_test: np.array) -> float:
    assert len(y_test.shape) == 1, "y_test should be 1D"

    y_pred = model.predict(X_test)
    return roc_auc_score(y_test, y_pred)


def oversample(X: np.ndarray, y: np.array) -> tuple[np.ndarray, np.array]:
    smote = SMOTE()
    return smote.fit_resample(X, y)


def normalize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaler = MinMaxScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test)
    return train, test
