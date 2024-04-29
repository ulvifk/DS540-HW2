from itertools import product

import numpy as np
from sklearn.model_selection import KFold, train_test_split

from src import oversample, normalize
from src.problem_data.problem_data import ProblemData
from tqdm import tqdm


class GridSearch:
    _model: type
    _problem_data: ProblemData
    _search_space: list[dict]
    _error_metric: callable
    _params_scores: list
    best_params: tuple[dict, float] = None


    def __init__(self, model: type, problem_data: ProblemData, search_space: dict, error_metric: callable):
        self._model = model
        self._problem_data = problem_data
        self._error_metric = error_metric
        self._search_space = [dict(zip(search_space.keys(), values)) for values in product(*search_space.values())]
        self._search_space.append({})

        self._search()

    def _search(self):
        self._params_scores = []

        for params in tqdm(self._search_space):
            folds = (KFold(n_splits=5, shuffle=True)
                     .split(self._problem_data.df_X, self._problem_data.df_Y))

            scores = []
            for train_indices, val_indices in folds:
                model = self._model(parameters=params)
                train_X, train_Y = (self._problem_data.df_X.values[train_indices],
                                    self._problem_data.df_Y.values[train_indices])
                train_Y = train_Y.reshape(-1)

                val_X, val_Y = (self._problem_data.df_X.values[val_indices],
                                self._problem_data.df_Y.values[val_indices])
                val_Y = val_Y.reshape(-1)

                train_X, test_X = normalize(train_X, val_X)
                train_X, train_Y = oversample(train_X, train_Y)

                model.fit(train_X, train_Y)

                scores.append(self._error_metric(model, val_X, val_Y))

            self._params_scores.append((params, np.mean(scores)))

        self.best_params = max(self._params_scores, key=lambda x: x[1])