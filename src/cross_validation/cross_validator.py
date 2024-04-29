import numpy as np

from src import oversample, XGBoost, normalize
from src.problem_data import ProblemData
from sklearn.model_selection import KFold
from tqdm import tqdm


class CrossValidator:
    _model_names: list[type]
    _problem_data: ProblemData
    _n_splits: int
    scores: dict[type, list[np.array]]
    _error_metric: callable
    best_model: type

    def __init__(self, model_names: list[type], problem_data: ProblemData, n_splits: int, error_metric: callable):
        self._model_names = model_names
        self._problem_data = problem_data
        self._n_splits = n_splits
        self._error_metric = error_metric
        self.scores = {}

        self._cross_validate()

    def _cross_validate(self):
        data_sets = list(KFold(n_splits=self._n_splits, shuffle=True, random_state=0)
                         .split(self._problem_data.df_X, self._problem_data.df_Y))

        p_bar = tqdm(self._model_names)
        for model_name in p_bar:
            p_bar.set_description(f"Testing {model_name.__name__}")
            self.scores[model_name] = []
            for train_indices, test_indices in data_sets:
                self.scores[model_name].append(self._test_and_return_score(model_name, train_indices, test_indices))

        avg_scores = {model_name: np.mean(scores) for model_name, scores in self.scores.items()}

        self.best_model = max(avg_scores, key=avg_scores.get)

        for model_name, score in avg_scores.items():
            print(f"{model_name.__name__}: {score}")

    def _test_and_return_score(self, model_name: type, train_indices: np.array, test_indices: np.array) -> np.float64:
        train_X, train_Y = (self._problem_data.df_X.values[train_indices],
                            self._problem_data.df_Y.values[train_indices])
        train_Y = train_Y.reshape(-1)

        test_X, test_Y = (self._problem_data.df_X.values[test_indices],
                          self._problem_data.df_Y.values[test_indices])
        test_Y = test_Y.reshape(-1)

        train_X, test_X = normalize(train_X, test_X)
        train_X, train_Y = oversample(train_X, train_Y)

        model = model_name()
        model.fit(train_X, train_Y)

        return self._error_metric(model, test_X, test_Y)
